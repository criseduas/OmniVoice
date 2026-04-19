import os
import sys
import threading
import time
import ctypes
from pathlib import Path

import wx
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F_audio
import sounddevice as sd
import numpy as np

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------- OPTIMIZACIONES NATIVAS PARA RTX 4070 ----------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True 
else:
    raise RuntimeError("CUDA no disponible. Esta GUI requiere una GPU NVIDIA.")

# ---------------- CONFIG ----------------
DEFAULT_MODEL = "k2-fsa/OmniVoice"
DEVICE = "cuda"

BASE_DIR = Path(__file__).parent.resolve()
VOICES_DIR = BASE_DIR / "voces"
AUDIOS_DIR = BASE_DIR / "audios"

os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(AUDIOS_DIR, exist_ok=True)

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
ALL_LANGS = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

# ---------------- CLASE NVDA MEJORADA ----------------
class NVDAController:
    def __init__(self):
        self.dll = None
        self.is_running = False
        
        if not sys.platform.startswith("win"):
            return

        possible_dlls = [
            BASE_DIR / "nvdaControllerClient64.dll",
            BASE_DIR / "nvdaControllerClient.dll"
        ]

        for dll_path in possible_dlls:
            if dll_path.exists():
                try:
                    self.dll = ctypes.windll.LoadLibrary(str(dll_path))
                    if self.dll.nvdaController_testIfRunning() == 0:
                        self.is_running = True
                        break
                except Exception:
                    pass

    def speak(self, text, interrupt=False):
        if self.is_running and self.dll:
            try:
                if interrupt:
                    self.dll.nvdaController_cancelSpeech()
                self.dll.nvdaController_speakText(ctypes.c_wchar_p(text))
            except Exception:
                pass

nvda = NVDAController()

# ---------------- UTIL ----------------
def read_txt_safe(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except:
        return ""

# ---------------- GUI ----------------
class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="OmniVoice GUI - Clonación de Voz", size=(900, 850))
        self.panel = wx.Panel(self)

        self.model = None
        self.model_loaded = False
        
        # SISTEMA DE CACHÉ PARA AUDIOS DE REFERENCIA
        self.cached_prompt = None
        self.cached_prompt_hash = ""

        self.build_ui()
        self.scan_voices()
        
        if nvda.is_running:
            self.log_msg("🟢 NVDA conectado correctamente.")
        else:
            self.log_msg("🟡 NVDA NO conectado. Asegúrate de tener 'nvdaControllerClient64.dll' en la carpeta.")

        self.load_model_async()

    def build_ui(self):
        main = wx.BoxSizer(wx.VERTICAL)

        self.status = wx.StaticText(self.panel, label="Iniciando motor de OmniVoice...")
        self.status.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        main.Add(self.status, 0, wx.ALL, 10)

        # VOZ
        main.Add(wx.StaticText(self.panel, label="Carpeta de voz (Speaker)"))
        self.voice_choice = wx.Choice(self.panel)
        self.voice_choice.Bind(wx.EVT_CHOICE, self.on_voice)
        main.Add(self.voice_choice, 0, wx.EXPAND | wx.ALL, 5)

        main.Add(wx.StaticText(self.panel, label="Archivo de audio de referencia (IDEAL: 10 a 15 segundos máximo)"))
        self.variant_choice = wx.Choice(self.panel)
        self.variant_choice.Bind(wx.EVT_CHOICE, self.on_variant)
        main.Add(self.variant_choice, 0, wx.EXPAND | wx.ALL, 5)

        main.Add(wx.StaticText(self.panel, label="Texto del audio de referencia (OBLIGATORIO)"))
        self.ref_text = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE)
        main.Add(self.ref_text, 0, wx.EXPAND | wx.ALL, 5)

        # TEXTO A SINTETIZAR
        main.Add(wx.StaticText(self.panel, label="Texto a sintetizar"))
        self.text = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE)
        main.Add(self.text, 1, wx.EXPAND | wx.ALL, 10)

        # IDIOMA
        main.Add(wx.StaticText(self.panel, label="Idioma destino"))
        self.lang = wx.Choice(self.panel, choices=ALL_LANGS)

        for i, lang in enumerate(ALL_LANGS):
            if "Spanish" in lang:
                self.lang.SetSelection(i)
                break

        main.Add(self.lang, 0, wx.EXPAND | wx.ALL, 5)


        main.Add(wx.StaticText(self.panel, label="Speed (1.0 = Normal)"))
        self.speed = wx.SpinCtrlDouble(self.panel, value="1.0", min=0.5, max=2.0, inc=0.1)
        self.bind_accessible(self.speed, "Speed")
        main.Add(self.speed, 0, wx.ALL, 5)

        # --- PARÁMETROS DINÁMICOS (desde omnivoice.py) ---
        self.config_controls = {}

        # Importar dataclasses para introspección
        import dataclasses
        config_fields = dataclasses.fields(OmniVoiceGenerationConfig)

        for field in config_fields:
            field_name = field.name
            default_val = field.default
            
            # Etiqueta
            label = f"{field_name.replace(chr(95), ' ').title()} (Default: {default_val})"
            
            if field.type == bool:
                ctrl = wx.CheckBox(self.panel, label=label)
                ctrl.SetValue(bool(default_val))
            elif field.type in (int, float):
                # Definir rangos genéricos seguros
                if isinstance(default_val, int):
                    ctrl = wx.SpinCtrlDouble(self.panel, value=str(default_val), min=0.0, max=1000.0, inc=1.0)
                    ctrl.SetDigits(0)
                else:
                    ctrl = wx.SpinCtrlDouble(self.panel, value=str(default_val), min=0.0, max=10.0, inc=0.01)
                    ctrl.SetDigits(2)
            else:
                continue # Salta tipos no soportados (como listas)

            self.bind_accessible(ctrl, field_name)
            main.Add(ctrl, 0, wx.ALL, 5)
            self.config_controls[field_name] = ctrl
        # ----------------------------------------------

        # Parámetros de inferencia (no están en Config pero sí en OmniVoice)
        main.Add(wx.StaticText(self.panel, label="Steps (32 es ideal)"))
        self.steps = wx.SpinCtrl(self.panel, value="32", min=4, max=64)
        self.bind_accessible(self.steps, "Pasos de inferencia")
        main.Add(self.steps, 0, wx.ALL, 5)

        main.Add(wx.StaticText(self.panel, label="Guidance scale (CFG. Default: 2.0)"))
        self.guidance = wx.SpinCtrlDouble(self.panel, value="2.0", min=0, max=5, inc=0.1)
        self.bind_accessible(self.guidance, "Guidance scale")
        main.Add(self.guidance, 0, wx.ALL, 5)

        # BOTÓN GENERAR
        main.Add(wx.StaticText(self.panel, label=""))
        self.btn = wx.Button(self.panel, label="Generar audio", size=(-1, 40))
        self.btn.Bind(wx.EVT_BUTTON, self.on_generate)
        main.Add(self.btn, 0, wx.ALL, 5)
        
        # LOG
        self.log = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        main.Add(self.log, 1, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(main)

    # -------- ACCESIBILIDAD --------
    def bind_accessible(self, ctrl, label):
        def on_focus(event):
            try:
                val = ctrl.GetValue()
            except:
                val = ""
            nvda.speak(f"{label} {val}", True)
            event.Skip()
        ctrl.Bind(wx.EVT_SET_FOCUS, on_focus)

    # -------- LOG --------
    def log_msg(self, msg):
        wx.CallAfter(self.log.AppendText, msg + "\n")
        wx.CallAfter(self.log.SetInsertionPointEnd)

    # -------- MODEL --------
    def load_model_async(self):
        def run():
            try:
                self.log_msg("Cargando modelo OmniVoice puro en VRAM (Sin Whisper/ASR)...")
                start_load = time.time()
                
                self.model = OmniVoice.from_pretrained(
                    DEFAULT_MODEL,
                    device_map={"":"cuda:0"}, 
                    dtype=torch.float16,
                    load_asr=False 
                )
                
                # Asegurar que el Tokenizador esté en la GPU (por si HF falla)
                if hasattr(self.model, "audio_tokenizer"):
                    self.model.audio_tokenizer.to(DEVICE)
                    self.model.audio_tokenizer.eval()

                self.log_msg(f"Modelo cargado en {time.time() - start_load:.2f}s. Calentando GPU...")
                
                # Warmup CON inference_mode (como debe ser)
                with torch.inference_mode():
                    _ = self.model.generate(
                        text="hola",
                        generation_config=OmniVoiceGenerationConfig(num_step=4)
                    )

                self.model_loaded = True
                wx.CallAfter(self.status.SetLabel, "Modelo listo - Listo para clonar")
                wx.CallAfter(self.status.SetForegroundColour, wx.Colour(0, 150, 0))
                wx.CallAfter(self.btn.Enable)
                self.log_msg("¡Sistema completamente listo!")
                nvda.speak("Modelo listo", True)

            except Exception as e:
                wx.CallAfter(self.status.SetLabel, "Error al cargar el modelo")
                wx.CallAfter(self.status.SetForegroundColour, wx.Colour(255, 0, 0))
                wx.CallAfter(self.log_msg, f"ERROR CRÍTICO: {str(e)}")

        threading.Thread(target=run, daemon=True).start()

    # -------- VOICES --------
    def scan_voices(self):
        voices = [d.name for d in VOICES_DIR.iterdir() if d.is_dir()]
        self.voice_choice.SetItems(voices)
        if voices:
            self.voice_choice.SetSelection(0)
            self.on_voice(None)

    def on_voice(self, e):
        voice = self.voice_choice.GetStringSelection()
        if e is not None:
            nvda.speak(f"Carpeta {voice}")

        path = VOICES_DIR / voice
        files = []
        for ext in AUDIO_EXTENSIONS:
            files.extend(path.glob(f"*{ext}"))

        names = [f.name for f in files]
        self.variant_choice.SetItems(names)

        if names:
            self.variant_choice.SetSelection(0)
            self.on_variant(None)

    def on_variant(self, e):
        voice = self.voice_choice.GetStringSelection()
        var = self.variant_choice.GetStringSelection()

        if e is not None:
            nvda.speak(f"Archivo {var}")

        txt = VOICES_DIR / voice / (Path(var).stem + ".txt")
        if txt.exists():
            self.ref_text.SetValue(read_txt_safe(txt))
        else:
            self.ref_text.SetValue("")

    # -------- GENERATE --------
    def on_generate(self, e):
        if not self.model_loaded:
            nvda.speak("Modelo no listo", True)
            return

        text = self.text.GetValue().strip()
        if not text:
            nvda.speak("Por favor, introduce el texto a sintetizar", True)
            return

        voice = self.voice_choice.GetStringSelection()
        var = self.variant_choice.GetStringSelection()
        if not voice or not var:
            nvda.speak("Selecciona un audio de referencia válido", True)
            return

        ref_audio = str(VOICES_DIR / voice / var)
        ref_text = self.ref_text.GetValue().strip()
        
        if not ref_text:
            nvda.speak("Falta el texto de referencia. Es obligatorio.", True)
            self.log_msg("❌ ERROR: El cuadro 'Texto del audio de referencia' está vacío.")
            return

        self.btn.Disable()
        self.btn.SetLabel("Generando... Espera")
        
        # Collect dynamic config values
        gen_kwargs = {"speed": self.speed.GetValue(), "lang": self.lang.GetStringSelection(), "steps": self.steps.GetValue(), "guidance_scale": self.guidance.GetValue()}
        
        for field_name, ctrl in self.config_controls.items():
            if isinstance(ctrl, wx.CheckBox):
                gen_kwargs[field_name] = ctrl.GetValue()
            elif isinstance(ctrl, wx.SpinCtrlDouble):
                val = ctrl.GetValue()
                # Determine type from config
                import dataclasses
                field = next(f for f in dataclasses.fields(OmniVoiceGenerationConfig) if f.name == field_name)
                if field.type == int:
                    val = int(val)
                elif field.type == float:
                    val = float(val)
                gen_kwargs[field_name] = val
        
        settings = {
            "text": text,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            **gen_kwargs
        }

        threading.Thread(
            target=self.run_gen,
            args=(settings,),
            daemon=True
        ).start()

    def run_gen(self, settings):
        try:
            nvda.speak("Generando audio. Por favor espera.", True)
            self.log_msg("-" * 40)
            self.log_msg(f"Iniciando proceso para: {Path(settings['ref_audio']).name}")

            current_prompt_hash = f"{settings['ref_audio']}_{settings['ref_text']}_{settings['preprocess_prompt']}"

            # FASE 1: LECTURA Y PROMPT (AHORA CON INFERENCE_MODE)
            if self.cached_prompt_hash == current_prompt_hash and self.cached_prompt is not None:
                self.log_msg("⚡ ¡CACHÉ ACTIVADO! Reutilizando características de voz del audio anterior...")
                prompt = self.cached_prompt
                t_prompt_elapsed = 0.0
            else:
                t_prompt_start = time.time()
                
                # --- MICRO-CRONÓMETROS ---
                t0 = time.time()
                wav_tensor, sr = torchaudio.load(settings['ref_audio'])
                self.log_msg(f"  -> Audio cargado del disco en {time.time()-t0:.3f}s")
                
                t0 = time.time()
                target_sr = self.model.sampling_rate
                if sr != target_sr:
                    wav_tensor = F_audio.resample(wav_tensor, sr, target_sr)
                if wav_tensor.shape[0] > 1:
                    wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
                self.log_msg(f"  -> Audio remuestreado a {target_sr}Hz en {time.time()-t0:.3f}s")

                t0 = time.time()
                
                # ¡LA SOLUCIÓN DEFINITIVA! Apagar el cálculo de gradientes.
                with torch.inference_mode():
                    prompt = self.model.create_voice_clone_prompt(
                        ref_audio=(wav_tensor, target_sr),
                        ref_text=settings['ref_text'],
                        preprocess_prompt=settings['preprocess_prompt']
                    )
                self.log_msg(f"  -> Tokenización en GPU (Higgs) SIN GRADIENTES completada en {time.time()-t0:.3f}s")
                
                t_prompt_elapsed = time.time() - t_prompt_start
                self.log_msg(f"✔️ Extracción de características TOTAL lista en {t_prompt_elapsed:.3f}s")
                
                self.cached_prompt = prompt
                self.cached_prompt_hash = current_prompt_hash

            # FASE 2: SÍNTESIS
            # Construir el diccionario solo con los campos de la configuración
            # Mapear "steps" a "num_step"
            config_kwargs = {k: v for k, v in settings.items() if k not in ("text", "ref_audio", "ref_text", "speed", "lang", "steps")}
            config_kwargs["num_step"] = settings["steps"]
            config = OmniVoiceGenerationConfig(**config_kwargs)

            lang = settings['lang']
            lang = None if lang == "Auto" else lang

            kwargs = dict(
                text=settings['text'],
                voice_clone_prompt=prompt,
                generation_config=config,
                language=lang
            )

            if settings['speed'] != 1.0:
                kwargs["speed"] = settings['speed']

            self.log_msg(f"Iniciando síntesis en GPU (Steps: {settings['steps']})...")
            t_gen_start = time.time()
            
            torch.cuda.empty_cache()
            
            with torch.inference_mode():
                audio_output = self.model.generate(**kwargs)
            
            t_gen_elapsed = time.time() - t_gen_start
            self.log_msg(f"✔️ Síntesis completada en {t_gen_elapsed:.3f}s")
            
            # FASE 3: GUARDADO Y REPRODUCCIÓN
            wave = audio_output[0] 
            sr = self.model.sampling_rate 
            
            wave_int16 = (wave * 32767).clip(-32768, 32767).astype(np.int16)

            filename = f"out_{int(time.time())}.wav"
            out_path = AUDIOS_DIR / filename
            sf.write(out_path, wave_int16, sr)

            total_time = t_prompt_elapsed + t_gen_elapsed
            self.log_msg(f"✅ ¡Éxito! Tiempo TOTAL: {total_time:.3f}s. Guardado: {filename}")

            sd.play(wave, sr)
            sd.wait() 

            nvda.speak("Audio generado y reproducido.", True)

        except Exception as e:
            wx.CallAfter(self.log_msg, f"❌ ERROR DURANTE LA GENERACIÓN: {str(e)}")
            nvda.speak("Ocurrió un error al generar", True)
        finally:
            wx.CallAfter(self.btn.Enable)
            wx.CallAfter(self.btn.SetLabel, "Generar audio")

if __name__ == "__main__":
    app = wx.App()
    MainFrame().Show()
    app.MainLoop()


