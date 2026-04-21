import os
import sys
import threading
import time
import ctypes
from pathlib import Path

import wx
import soundfile as sf
import torch
import sounddevice as sd
import numpy as np

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------- OPTIMIZACIONES NATIVAS PARA RTX 4070 ----------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
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

# ---------------- NVDA ----------------
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

def read_txt_safe(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except:
        return ""

# ---------------- GUI PRINCIPAL ----------------
class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="OmniVoice GUI - Clonación de Voz", size=(950, 900))
        self.panel = wx.Panel(self)
        self.model = None
        self.model_loaded = False
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
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Barra de estado
        self.status = wx.StaticText(self.panel, label="Iniciando motor de OmniVoice...")
        self.status.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        main_sizer.Add(self.status, 0, wx.ALL, 10)

        # --- Sección de voz y referencia (fija) ---
        main_sizer.Add(wx.StaticText(self.panel, label="Carpeta de voz (Speaker)"), 0, wx.ALL, 5)
        self.voice_choice = wx.Choice(self.panel)
        self.voice_choice.Bind(wx.EVT_CHOICE, self.on_voice)
        main_sizer.Add(self.voice_choice, 0, wx.EXPAND | wx.ALL, 5)

        main_sizer.Add(wx.StaticText(self.panel, label="Archivo de audio de referencia (ideal 10-15s)"), 0, wx.ALL, 5)
        self.variant_choice = wx.Choice(self.panel)
        self.variant_choice.Bind(wx.EVT_CHOICE, self.on_variant)
        main_sizer.Add(self.variant_choice, 0, wx.EXPAND | wx.ALL, 5)

        main_sizer.Add(wx.StaticText(self.panel, label="Texto del audio de referencia (OBLIGATORIO)"), 0, wx.ALL, 5)
        self.ref_text = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE, size=(-1, 60))
        main_sizer.Add(self.ref_text, 0, wx.EXPAND | wx.ALL, 5)

        # --- Texto a sintetizar ---
        main_sizer.Add(wx.StaticText(self.panel, label="Texto a sintetizar"), 0, wx.ALL, 5)
        self.text = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE, size=(-1, 100))
        main_sizer.Add(self.text, 1, wx.EXPAND | wx.ALL, 10)

        # --- Parámetros generales ---
        general_box = wx.StaticBox(self.panel, label="Parámetros generales")
        general_sizer = wx.StaticBoxSizer(general_box, wx.VERTICAL)

        # Idioma
        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lang_sizer.Add(wx.StaticText(self.panel, label="Idioma destino:"), 0, wx.ALL | wx.CENTER, 5)
        self.lang = wx.Choice(self.panel, choices=ALL_LANGS)
        for i, lang in enumerate(ALL_LANGS):
            if "Spanish" in lang:
                self.lang.SetSelection(i)
                break
        lang_sizer.Add(self.lang, 1, wx.EXPAND | wx.ALL, 5)
        general_sizer.Add(lang_sizer, 0, wx.EXPAND)

        # Velocidad
        speed_sizer = wx.BoxSizer(wx.HORIZONTAL)
        speed_sizer.Add(wx.StaticText(self.panel, label="Speed (1.0 = normal):"), 0, wx.ALL | wx.CENTER, 5)
        self.speed = wx.SpinCtrlDouble(self.panel, value="1.0", min=0.5, max=2.0, inc=0.1)
        speed_sizer.Add(self.speed, 1, wx.EXPAND | wx.ALL, 5)
        general_sizer.Add(speed_sizer, 0, wx.EXPAND)

        main_sizer.Add(general_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # --- Parámetros dinámicos de OmniVoiceGenerationConfig (excluyendo chunking) ---
        config_scroll = wx.ScrolledWindow(self.panel)
        config_scroll.SetScrollRate(0, 10)
        config_sizer = wx.BoxSizer(wx.VERTICAL)

        config_box = wx.StaticBox(config_scroll, label="Parámetros de generación (OmniVoiceGenerationConfig)")
        config_inner = wx.StaticBoxSizer(config_box, wx.VERTICAL)

        import dataclasses
        # Excluir parámetros de chunking (son internos del modelo)
        EXCLUDED_FIELDS = {'audio_chunk_duration', 'audio_chunk_threshold'}
        config_fields = [f for f in dataclasses.fields(OmniVoiceGenerationConfig) if f.name not in EXCLUDED_FIELDS]

        self.config_controls = {}
        for field in config_fields:
            field_name = field.name
            default_val = field.default
            label_text = f"{field_name.replace('_', ' ').title()} (default: {default_val})"

            if field.type == bool:
                ctrl = wx.CheckBox(config_scroll, label=label_text)
                ctrl.SetValue(bool(default_val))
            elif field.type in (int, float):
                if isinstance(default_val, int):
                    ctrl = wx.SpinCtrl(config_scroll, value=str(default_val), min=0, max=1000)
                else:
                    ctrl = wx.SpinCtrlDouble(config_scroll, value=str(default_val), min=0.0, max=200.0, inc=0.5)
                    ctrl.SetDigits(1)
            else:
                continue

            self.bind_accessible(ctrl, field_name)
            config_inner.Add(ctrl, 0, wx.EXPAND | wx.ALL, 5)
            self.config_controls[field_name] = ctrl

        config_sizer.Add(config_inner, 0, wx.EXPAND)
        config_scroll.SetSizer(config_sizer)
        config_scroll.SetMinSize((-1, 250))
        main_sizer.Add(config_scroll, 0, wx.EXPAND | wx.ALL, 5)

        # --- Botón de generación ---
        self.btn = wx.Button(self.panel, label="Generar audio", size=(-1, 40))
        self.btn.Bind(wx.EVT_BUTTON, self.on_generate)
        main_sizer.Add(self.btn, 0, wx.ALL | wx.CENTER, 10)

        # --- Log de mensajes ---
        self.log = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        main_sizer.Add(self.log, 1, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(main_sizer)

    def bind_accessible(self, ctrl, label):
        def on_focus(event):
            try:
                val = ctrl.GetValue()
            except:
                val = ""
            nvda.speak(f"{label} {val}", True)
            event.Skip()
        ctrl.Bind(wx.EVT_SET_FOCUS, on_focus)

    def log_msg(self, msg):
        wx.CallAfter(self.log.AppendText, msg + "\n")
        wx.CallAfter(self.log.SetInsertionPointEnd)

    # -------- MODELO --------
    def load_model_async(self):
        def run():
            try:
                self.log_msg("Cargando modelo OmniVoice (sin ASR)...")
                start_load = time.time()
                self.model = OmniVoice.from_pretrained(
                    DEFAULT_MODEL,
                    device_map="cuda",
                    torch_dtype=torch.float16,
                    load_asr=False
                )
                self.log_msg(f"Modelo cargado en {time.time()-start_load:.2f}s. Calentando GPU...")
                # Warmup simple y rápido como el original
                with torch.inference_mode():
                    _ = self.model.generate(
                        text="hola",
                        generation_config=OmniVoiceGenerationConfig(num_step=4)
                    )
                torch.cuda.synchronize()
                self.model_loaded = True
                wx.CallAfter(self.status.SetLabel, "Modelo listo - Listo para clonar")
                wx.CallAfter(self.status.SetForegroundColour, wx.Colour(0,150,0))
                wx.CallAfter(self.btn.Enable)
                self.log_msg("¡Sistema completamente listo!")
                nvda.speak("Modelo listo", True)
            except Exception as e:
                wx.CallAfter(self.status.SetLabel, "Error al cargar el modelo")
                wx.CallAfter(self.status.SetForegroundColour, wx.Colour(255,0,0))
                wx.CallAfter(self.log_msg, f"ERROR CRÍTICO: {str(e)}")
        threading.Thread(target=run, daemon=True).start()

    # -------- VOZ Y REFERENCIA --------
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

    # -------- GENERACIÓN --------
    def on_generate(self, e):
        if not self.model_loaded:
            nvda.speak("Modelo no listo", True)
            return

        text = self.text.GetValue().strip()
        if not text:
            nvda.speak("Introduce el texto a sintetizar", True)
            return

        voice = self.voice_choice.GetStringSelection()
        var = self.variant_choice.GetStringSelection()
        if not voice or not var:
            nvda.speak("Selecciona un audio de referencia", True)
            return

        ref_audio = str(VOICES_DIR / voice / var)
        ref_text = self.ref_text.GetValue().strip()
        if not ref_text:
            nvda.speak("El texto de referencia es obligatorio", True)
            self.log_msg("❌ ERROR: El cuadro 'Texto del audio de referencia' está vacío.")
            return

        # Recolectar valores de los controles dinámicos
        config_kwargs = {}
        for field_name, ctrl in self.config_controls.items():
            if isinstance(ctrl, wx.CheckBox):
                config_kwargs[field_name] = ctrl.GetValue()
            elif isinstance(ctrl, wx.SpinCtrl):
                config_kwargs[field_name] = int(ctrl.GetValue())
            elif isinstance(ctrl, wx.SpinCtrlDouble):
                config_kwargs[field_name] = float(ctrl.GetValue())

        language = self.lang.GetStringSelection()
        language = None if language == "Auto" else language
        speed_val = self.speed.GetValue()

        self.btn.Disable()
        self.btn.SetLabel("Generando... Espera")

        threading.Thread(
            target=self.run_gen_with_retry,
            args=(text, ref_audio, ref_text, language, speed_val, config_kwargs),
            daemon=True
        ).start()

    def run_gen_with_retry(self, text, ref_audio, ref_text, language, speed_val, config_kwargs):
        try:
            self.run_gen(text, ref_audio, ref_text, language, speed_val, config_kwargs)
        except torch.cuda.OutOfMemoryError as e:
            self.log_msg(f"⚠️ Error de memoria: {str(e)}. Reintentando con chunking más pequeño...")
            torch.cuda.empty_cache()
            # Reducir temporalmente audio_chunk_threshold (parámetro interno)
            original_threshold = config_kwargs.get("audio_chunk_threshold", 30.0)
            config_kwargs["audio_chunk_threshold"] = 15.0
            try:
                self.run_gen(text, ref_audio, ref_text, language, speed_val, config_kwargs)
            except Exception as e2:
                self.log_msg(f"❌ También falló con chunking pequeño: {str(e2)}")
                nvda.speak("Error persistente de memoria", True)
            finally:
                config_kwargs["audio_chunk_threshold"] = original_threshold
        except Exception as e:
            self.log_msg(f"❌ ERROR: {str(e)}")
            nvda.speak("Error durante la generación", True)
        finally:
            wx.CallAfter(self.btn.Enable)
            wx.CallAfter(self.btn.SetLabel, "Generar audio")
            # Limpiar caché CUDA después de cada generación
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def run_gen(self, text, ref_audio, ref_text, language, speed_val, config_kwargs):
        nvda.speak("Generando audio. Por favor espera.", True)
        self.log_msg("-" * 40)
        self.log_msg(f"Iniciando para: {Path(ref_audio).name}")

        # Construir hash para caché (incluye preprocess_prompt del config)
        pp = config_kwargs.get("preprocess_prompt", True)
        current_hash = f"{ref_audio}_{ref_text}_{pp}"

        # ----- FASE 1: Crear o recuperar prompt de clonación -----
        if self.cached_prompt_hash == current_hash and self.cached_prompt is not None:
            self.log_msg("⚡ Usando caché de características de voz")
            prompt = self.cached_prompt
            t_prompt = 0.0
        else:
            t_start = time.time()
            with torch.inference_mode():
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    preprocess_prompt=pp
                )
            t_prompt = time.time() - t_start
            self.log_msg(f"✔️ Prompt creado en {t_prompt:.2f}s")
            self.cached_prompt = prompt
            self.cached_prompt_hash = current_hash

        # ----- FASE 2: Generar audio -----
        gen_config = OmniVoiceGenerationConfig(**config_kwargs)

        self.log_msg(f"Iniciando síntesis (steps={gen_config.num_step})...")
        t_gen = time.time()
        with torch.inference_mode():
            audio_output = self.model.generate(
                text=text,
                voice_clone_prompt=prompt,
                generation_config=gen_config,
                language=language,
                speed=speed_val,
            )
        t_gen = time.time() - t_gen
        self.log_msg(f"✔️ Síntesis completada en {t_gen:.2f}s")

        # ----- FASE 3: Guardar y reproducir (conversión eficiente) -----
        wave = audio_output[0]
        sr_out = self.model.sampling_rate
        wave_int16 = (wave * 32767).clip(-32768, 32767).astype(np.int16)

        filename = f"out_{int(time.time())}.wav"
        out_path = AUDIOS_DIR / filename
        sf.write(out_path, wave_int16, sr_out)

        total = t_prompt + t_gen
        self.log_msg(f"✅ Éxito! Tiempo TOTAL: {total:.2f}s -> {filename}")
        sd.play(wave, sr_out)
        sd.wait()
        nvda.speak("Audio generado y reproducido.", True)

if __name__ == "__main__":
    app = wx.App()
    MainFrame().Show()
    app.MainLoop()