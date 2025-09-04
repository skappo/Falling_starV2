import argparse
import cv2
import numpy as np
import time
import threading
import os
import subprocess
import logging
import json
import sys
import importlib
import shutil
import queue
from datetime import datetime
from collections import deque
from picamera2 import Picamera2
from libcamera import controls
from colorama import Fore, Style, Back, init

# --- Global Configuration Variable ---
# This dictionary will replace the `args` object and hold all runtime settings.
APP_CONFIG = {}

# === CONTROLLO DIPENDENZE ===
# Blocco di controllo per garantire che tutte le librerie necessarie siano installate
# prima di avviare lo script. Fornisce messaggi chiari e comandi di installazione.
REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "picamera2": "picamera2",
    "ffmpeg": None,
    "colorama": "colorama"
}

missing = []

for module_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        if module_name == "ffmpeg":
            if not shutil.which("ffmpeg"):
                raise ImportError
        else:
            importlib.import_module(module_name)
    except ImportError:
        missing.append(pip_name or module_name)

if missing:
    print("\n🚫 Dipendenze mancanti rilevate:")
    for pkg in missing:
        print(f"  - {pkg}")
    print("\n➡️  Puoi installarle con:")
    print(f"  pip install {' '.join([pkg for pkg in missing if pkg])}")
    print("🔧 Oppure su Raspberry Pi:\n  sudo apt install python3-opencv python3-numpy python3-picamera2 ffmpeg\n")
    sys.exit(1)

# === RILEVAMENTO MODELLO PI ===
def detect_pi_model():
    try:
        with open("/proc/device-tree/model", "r") as f:
            model_info = f.read().lower()
            if "raspberry pi 5" in model_info: return "pi5"
            elif "raspberry pi 4" in model_info: return "pi4"
    except: pass
    return "pi4"

pi_model = detect_pi_model()

# === FUNZIONE PER SALVARE LA CONFIGURAZIONE ===
CONFIG_FILE = "config.json"
def save_config(config_dict):
    """Salva un dizionario di configurazione in formato JSON."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"{Fore.GREEN}[CONFIG] Configurazione salvata con successo in {CONFIG_FILE}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Impossibile salvare la configurazione in {CONFIG_FILE}: {e}{Style.RESET_ALL}")
        
# === EDITOR INTERATTIVO DELLA CONFIGURAZIONE (Now the main entry point) ===
def edit_config_interactive(current_config):
    """
    Mostra un menu interattivo per modificare la configurazione.
    Restituisce la configurazione finale se l'utente sceglie di avviare, altrimenti None.
    """
    cfg = current_config.copy()
    while True:
        print("\n" + "="*15 + " MENU CONFIGURAZIONE " + "="*15)
        
        # Stampa i parametri in modo leggibile
        keys = list(cfg.keys())
        for i, key in enumerate(keys, start=1):
            value = cfg[key]
            # Formattazione per una migliore leggibilità
            if value is None:
                value_str = f"{Fore.LIGHTBLACK_EX}Non impostato{Style.RESET_ALL}"
            elif isinstance(value, bool):
                value_str = f"{Fore.CYAN}{value}{Style.RESET_ALL}"
            else:
                value_str = f"{Fore.YELLOW}{value}{Style.RESET_ALL}"
            print(f"  {Style.BRIGHT}{i:2d}. {key:<25}{Style.RESET_ALL} = {value_str}")

        # Aggiunge l'opzione di avvio
        print(f"\n  {Style.BRIGHT}{Fore.GREEN} 0. Avvia Programma{Style.RESET_ALL}")
        choice = input(f"\nScegli un numero da modificare, {Fore.GREEN}0 per avviare{Style.RESET_ALL}, o lascia vuoto per uscire: ")

        if not choice.strip():
            return None # L'utente ha scelto di uscire
        
        if choice == "0":
            print("[INFO] Salvataggio della configurazione finale...")
            save_config(cfg)
            return cfg # Restituisce la config finale per avviare il programma

        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            key_to_edit = keys[int(choice) - 1]
            current_value = cfg[key_to_edit]
            new_val_str = input(f"Nuovo valore per '{key_to_edit}' (attuale: {current_value}): ")

            if not new_val_str.strip():
                print("[INFO] Nessuna modifica effettuata.")
                continue

            # Tenta di convertire il nuovo valore nel tipo corretto
            try:
                if isinstance(current_value, bool):
                    new_val = new_val_str.lower() in ['true', '1', 't', 'y']
                elif isinstance(current_value, int):
                    new_val = int(new_val_str)
                elif isinstance(current_value, float):
                    new_val = float(new_val_str)
                elif current_value is None: # Se il valore era None, prova a indovinare il tipo
                    if new_val_str.isdigit(): new_val = int(new_val_str)
                    elif new_val_str.replace(".", "", 1).isdigit(): new_val = float(new_val_str)
                    else: new_val = new_val_str
                else: # Altrimenti, mantieni come stringa
                    new_val = new_val_str
                
                cfg[key_to_edit] = new_val
                print(f"{Fore.GREEN}[OK] Valore aggiornato: {key_to_edit} = {new_val}{Style.RESET_ALL}")

            except (ValueError, TypeError):
                print(f"{Fore.RED}[ERRORE] Input non valido. Il tipo non è stato modificato.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}[ERRORE] Scelta non valida.{Style.RESET_ALL}")
         
# === LOGGING CLASS (Defined in Global Scope) ===
class ColoredFormatter(logging.Formatter):
    """Formatter personalizzato che colora i log in base allo stato o al livello."""
    STATE_COLORS = {"IDLE": "\x1b[90m", "METEOR_FINDER": "\x1b[94m", "TIMELAPSE": "\x1b[96m", "POST_PROCESSING": "\x1b[95m", "EVENT": "\x1b[92m", "SHUTDOWN": "\x1b[93m"}
    LEVEL_COLORS = {logging.WARNING: "\x1b[33;20m", logging.ERROR: "\x1b[31;20m", logging.CRITICAL: "\x1b[31;1m"}
    RESET = "\x1b[0m"
    def format(self, record):
        log_message = super().format(record)
        if hasattr(record, 'state') and record.state in self.STATE_COLORS:
            return f"{self.STATE_COLORS[record.state]}{log_message}{self.RESET}"
        return f"{self.LEVEL_COLORS.get(record.levelno, '')}{log_message}{self.RESET}"

# === NUOVA FUNZIONE: CREAZIONE VIDEO TIMELAPSE ===
def create_timelapse_video(image_folder, output_filename, fps, cleanup=False):
    # Utilizza ffmpeg per creare un video da una sequenza di immagini.
    logging.info(f"[POST-PROCESSING] Avvio creazione video timelapse da {image_folder}")
    
    # Trova e ordina tutte le immagini JPG nella cartella
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        logging.warning("[POST-PROCESSING] Nessuna immagine trovata per creare il video timelapse.")
        return

    # Crea un file di testo temporaneo che elenca i file di immagine per ffmpeg
    list_filename = os.path.join(image_folder, "imagelist.txt")
    with open(list_filename, 'w') as f:
        for img in images:
            f.write(f"file '{img}'\n")

    # Costruisce il comando ffmpeg
    # Questo comando è robusto e produce video di alta qualità
    cmd = [
        "ffmpeg",
        "-y",                             # Sovrascrive il file di output se esiste
        "-r", str(fps),                   # FPS del video in uscita
        "-f", "concat",                   # Usa il formato concatenatore
        "-safe", "0",
        "-i", list_filename,              # File di input con l'elenco delle immagini
        "-c:v", "libx264",                # Codec video H.264
        "-preset", "slow",                # Preset di qualità (slow = migliore compressione)
        "-crf", "18",                     # Fattore di qualità (più basso è meglio, 18 è quasi lossless)
        "-pix_fmt", "yuv420p",            # Formato pixel per la massima compatibilità
        output_filename
    ]

    try:
        # Esegui il comando ffmpeg
        subprocess.run(cmd, cwd=image_folder, check=True, capture_output=True, text=True)
        logging.info(f"[POST-PROCESSING] Video timelapse creato con successo: {output_filename}")

        # Pulisci i file temporanei e le immagini originali se richiesto
        os.remove(list_filename)
        if cleanup:
            logging.info("[POST-PROCESSING] Pulizia delle immagini JPG originali...")
            for img in images:
                os.remove(os.path.join(image_folder, img))
            logging.info("[POST-PROCESSING] Pulizia completata.")

    except subprocess.CalledProcessError as e:
        logging.error(f"[POST-PROCESSING] Errore durante la creazione del video timelapse con ffmpeg.")
        logging.error(f"  Comando: {' '.join(cmd)}")
        logging.error(f"  Output di ffmpeg:\n{e.stderr}")
    except Exception as e:
        logging.error(f"[POST-PROCESSING] Errore imprevisto: {e}")

# === FUNZIONI DI SCHEDULING ===
def is_time_in_interval(start_str, end_str):
    # Controlla se l'ora corrente rientra in un dato intervallo (gestisce la mezzanotte).
    if not start_str or not end_str:
        return False
    now = datetime.now().time()
    start = datetime.strptime(start_str, "%H:%M").time()
    end = datetime.strptime(end_str, "%H:%M").time()
    if start < end:
        return start <= now < end
    else: # Handles overnight interval
        return now >= start or now < end

# === RILEVAMENTO CAMERA CON AUTOFOCUS ===
def is_autofocus_camera(picam2_object):
    # Camera Info, if v3 disable autofocus
    try:
        model = picam2_object.camera_properties.get('Model', 'unknown').lower()
        if 'imx708' in model:
            logging.info(f"Rilevato sensore IMX708 (Camera Module 3). Verranno applicati i controlli di fuoco manuale per il timelapse.")
            return True
    except Exception as e:
        logging.error(f"Impossibile determinare il modello della camera: {e}")
    return False

# === SUPPORTO H264 PER RASPBERRY PI ===
def start_ffmpeg_writer(filename, width, height, framerate):
    """
    Costruisce e avvia il comando ffmpeg per la registrazione video.
    - Accetta input GRAYSCALE per mantenere semplice la pipeline Python.
    - Usa un filtro video (`-vf`) per convertire in YUV420p, il formato richiesto
      dall'encoder hardware `h264_v4l2m2m`.
    """
    cmd = [
        "ffmpeg", "-y", 
        "-f", "rawvideo", 
        "-pix_fmt", "gray",
        "-s", f"{width}x{height}",
        "-r", str(framerate), 
        "-i", "-", 
        "-vf", "format=yuv420p",
        "-an",
        "-vcodec", "h264_v4l2m2m", # <-- USA L'ENCODER HARDWARE del Raspberry Pi
        "-preset", "ultrafast", 
        "-crf", "23", 
        filename
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def update_perf_counter(counter_name, increment=1):
    # Funzione thread-safe per aggiornare i contatori di performance.
    global frames_captured, frames_processed, frames_written, events_triggered
    with lock_perf:
        if counter_name == "captured": frames_captured += increment
        elif counter_name == "processed": frames_processed += increment
        elif counter_name == "written": frames_written += increment
        elif counter_name == "events": events_triggered += increment

# === THREADS ===

# --- Universal Threads ---
def capture_thread_meteor(state_event, width, height):
    """
    THREAD CRITICO: Produttore per la modalità METEOR_FINDER.
    - Cattura due stream (alta e bassa risoluzione) dalla camera.
    - Usa il pattern `capture_request` per garantire dati validi e stabili.
    - Converte immediatamente entrambi gli stream in grayscale per efficienza.
    - Usa `put_nowait` per scartare frame se la pipeline è piena, evitando blocchi.
    """
    # Pre-calcola le dimensioni dello stream a bassa risoluzione per lo slicing
    lores_w = width // APP_CONFIG['downscale_factor']
    lores_h = height // APP_CONFIG['downscale_factor']
    
    while running and state_event.is_set():
        request = None # Inizializza a None per sicurezza
        try:
            # --- THIS IS THE DEFINITIVE FIX ---
            # 1. Cattura la richiesta per entrambi gli stream.
            request = picam2.capture_request()

            # 2. Estrae i dati in array NumPy garantiti.
            main_frame_yuv = request.make_array("main")
            lores_frame_yuv = request.make_array("lores")
           
            # Converte in grayscale QUI, una sola volta.
            main_frame_gray = main_frame_yuv[:height, :width]
            lores_frame_gray = lores_frame_yuv[:lores_h, :lores_w]

            # 3. Ora che abbiamo array NumPy, li mettiamo nella coda.
            #    La conversione in grayscale avverrà nel processing_thread.
            frame_queue.put_nowait((main_frame_gray, lores_frame_gray))
            update_perf_counter("captured")

        except queue.Full:
            pass # Comportamento previsto
        except Exception as e:
            logging.error(f"[CAPTURE] Errore imprevisto durante la cattura del frame: {e}")
        finally:
            # 4. Rilascia sempre la richiesta per liberare i buffer.
            if request:
                request.release()

def monitor_thread():
    """
    THREAD CRITICO: Stampa lo stato del sistema e i contatori.
    - Funziona in background per fornire un "heartbeat" e metriche di performance.
    - In modalità IDLE, stampa un messaggio semplificato.
    - Calcola i totali di sessione (persistenti) e i tassi attuali (FPS).
    """
    global frames_captured, frames_processed, frames_written, events_triggered, lock_perf
    INTERVAL = 15 # Aumentiamo l'intervallo per rendere i totali più significativi

    # Variabili per calcolare i tassi (FPS) tra un intervallo e l'altro
    last_fc, last_fp, last_fw, last_ev = 0, 0, 0, 0

    while running:
        time.sleep(INTERVAL)

        # Se lo script è inattivo, non stampare nulla e attendi il prossimo ciclo.
        if current_state == "IDLE":
            logging.info(f"[MONITOR] Stato @ {datetime.now().strftime('%H:%M:%S')}: IDLE. In attesa del prossimo task programmato.", extra={'state': 'IDLE'})
            continue
        
        # Legge i valori correnti dei contatori globali in modo thread-safe
        with lock_perf:
            current_fc = frames_captured
            current_fp = frames_processed
            current_fw = frames_written
            current_ev = events_triggered

        # --- Calcolo dei Tassi (Eventi per Intervallo) ---
        # Calcola quanti eventi si sono verificati dall'ultima stampa
        fc_interval = current_fc - last_fc
        fp_interval = current_fp - last_fp
        fw_interval = current_fw - last_fw
        
        # Aggiorna i valori "last" per il prossimo ciclo
        last_fc, last_fp, last_fw, last_ev = current_fc, current_fp, current_fw, current_ev

        # Calcola gli FPS basati solo sull'attività nell'ultimo intervallo
        fps_cap = fc_interval / INTERVAL
        fps_proc = fp_interval / INTERVAL
        fps_write = fw_interval / INTERVAL
        
        # --- Preparazione dell'Output ---
        status = "SÌ" if 'recording_event' in globals() and recording_event.is_set() else "NO"
        q_sizes = f"[{frame_queue.qsize()}/{output_queue.qsize()}/{snapshot_queue.qsize()}/{timelapse_writer_queue.qsize()}]"
        
        # --- Color helpers ---
        def color_state(state):
            mapping = {
                "METEOR_FINDER": Fore.BLUE + state + Style.RESET_ALL,
                "TIMELAPSE": Fore.CYAN + state + Style.RESET_ALL,
                "POST_PROCESSING": Fore.MAGENTA + state + Style.RESET_ALL,
                "IDLE": Fore.LIGHTBLACK_EX + state + Style.RESET_ALL,
            }
            return mapping.get(state, state)

        def color_queue(current, maximum):
            fill = current / maximum if maximum > 0 else 0
            if fill < 0.5: return Fore.GREEN + f"{current}" + Style.RESET_ALL
            elif fill < 0.8: return Fore.YELLOW + f"{current}" + Style.RESET_ALL
            else: return Fore.RED + f"{current}" + Style.RESET_ALL

        # Build colored queue strings
        q_frame = color_queue(frame_queue.qsize(), APP_CONFIG['frame_queue_maxsize'])
        q_out = color_queue(output_queue.qsize(), APP_CONFIG['output_queue_maxsize'])
        q_snap = color_queue(snapshot_queue.qsize(), APP_CONFIG['snapshot_queue_maxsize'])
        q_tl = color_queue(timelapse_writer_queue.qsize(), 10)

        # --- Output (same structure as before, but with colors) ---
        logging.info(f"[MONITOR] Stato @ {datetime.now().strftime('%H:%M:%S')}: {color_state(current_state)}")
        # Stampa i totali incrementali
        logging.info(f"  Totali Sessione -> Frame Acquisiti: {current_fc} | Frame Processati: {current_fp} | Frame Scritti: {current_fw}")
        logging.info(f"  Totali Sessione -> Eventi/Scatti Rilevati: {current_ev}")
        # Stampa i tassi attuali
        logging.info(f"  Tasso Attuale -> FPS (Acq/Proc/Scrit): {fps_cap:.1f}/{fps_proc:.1f}/{fps_write:.1f}")
        logging.info(f"  Stato -> Registrazione Attiva: {status}, Code [Meteora/Video/Snap/Timelapse]: "f"[{q_frame}/{q_out}/{q_snap}/{q_tl}]")

# --- Meteor Finder Threads ---
def processing_thread(state_event, effective_framerate, pre_event_buffer, width, height): 
    """
    THREAD CRITICO: Il "cervello" della modalità METEOR_FINDER.
    - Consumatore della `frame_queue`.
    - Esegue la logica di rilevamento sul frame a bassa risoluzione.
    - Gestisce il `pre_event_buffer` con i frame ad alta risoluzione.
    - Orchestra l'inizio di una registrazione.
    - Gestisce la terminazione di una registrazione entrando in uno stato `is_shutting_down`
      per garantire l'invio affidabile del segnale di stop (`None`) al writer.
    """
    global reference_frame, last_event_time, out, ffmpeg_proc, record_start_time
    
    scaled_trigger_area = APP_CONFIG['trigger_area'] / (APP_CONFIG['downscale_factor']**2)
    is_shutting_down = False
    
    while running and state_event.is_set():
        
        # --- Blocco di Arresto Prioritario ---
        # Questo blocco ha la priorità su tutto il resto. Se dobbiamo arrestare una
        # registrazione, ci concentriamo solo su quello.
        if is_shutting_down:
            try:
                output_queue.put(None, block=False) # Invia il segnale di stop
                logging.info("[PROCESS] Segnale di stop inviato con successo al writer.")
                
                # Resetta lo stato solo DOPO aver inviato il segnale con successo
                is_shutting_down = False
                pre_event_buffer.clear()
                recording_event.clear()
                
            except queue.Full:
                # La coda è ancora piena, il writer è indietro. Aspettiamo e riproviamo.
                time.sleep(0.5)
            
            # Torna all'inizio del loop per riprovare a inviare il segnale
            # o per uscire se lo stato del thread è cambiato.
            continue

        # --- Acquisizione Frame ---
        try:
            # Usa get_nowait per un loop reattivo che non si blocca.
            full_res_frame, detection_frame = frame_queue.get_nowait()
            update_perf_counter("processed")
        except queue.Empty:
            # È normale che la coda sia vuota, facciamo una breve pausa.
            time.sleep(0.01)
            continue
        
        # --- Logica di Rilevamento ---
        # Lavora sempre su frame in scala di grigi.
        blurred = cv2.medianBlur(detection_frame, 3)
        motion_detected = False

        if STRATEGY == "diff":
            if reference_frame is None:
                reference_frame = blurred.copy().astype("float")
                continue
            frame_diff = cv2.absdiff(blurred, cv2.convertScaleAbs(reference_frame))
            _, thresh = cv2.threshold(frame_diff, APP_CONFIG['diff_threshold'], 255, cv2.THRESH_BINARY)
            changed_area = cv2.countNonZero(thresh)
            motion_detected = changed_area > scaled_trigger_area
            cv2.accumulateWeighted(blurred, reference_frame, APP_CONFIG['learning_rate'])
        else: # contour strategy
            _, thresh = cv2.threshold(blurred, APP_CONFIG['min_brightness'], 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = any(cv2.contourArea(c) > APP_CONFIG['min_area'] for c in contours)

        # --- Gestione Buffer e Registrazione ---
        pre_event_buffer.append(full_res_frame.copy())
        current_time = time.time()

        # Caso 1: Rilevato un nuovo evento
        if motion_detected and not recording_event.is_set() and (current_time - last_event_time > 2):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(APP_CONFIG['output_dir'], f"evento_{timestamp}.{extension}")
            snapshot_path = os.path.join(APP_CONFIG['output_dir'], f"snapshot_{timestamp}.jpg")
            
            try:
                snapshot_queue.put((snapshot_path, full_res_frame.copy()), block=False)
            except queue.Full:
                logging.warning("[PROCESS] Coda snapshot piena, snapshot scartato.")            
            
            if APP_CONFIG['codec'] == "h264":
                ffmpeg_proc = start_ffmpeg_writer(filename, width, height, effective_framerate)
            else:
                out = cv2.VideoWriter(filename, fourcc, effective_framerate, (width, height), isColor=False)
            
            for f in pre_event_buffer:
                try:
                    output_queue.put(f, timeout=0.1)
                except queue.Full:
                    logging.warning("[PROCESS] Coda di output piena durante la scrittura del pre-evento.")

            recording_event.set()
            record_start_time = current_time
            last_event_time = current_time
            update_perf_counter("events")
            logging.info(f"[EVENT] Evento rilevato: registrazione iniziata {filename}")

        # Caso 2: La registrazione è già in corso
        elif recording_event.is_set():
            try:
                output_queue.put(full_res_frame, timeout=1)
            except queue.Full:
                logging.warning("[PROCESS] Coda di output piena, frame perso")

            # Controlla se è ora di terminare la registrazione
            if time.time() - record_start_time > APP_CONFIG['record_duration']:
                logging.info("[PROCESS] Durata registrazione terminata. Inizio procedura di arresto...")
                is_shutting_down = True

    # --- Uscita dal Loop Principale ---
    # Se il thread deve terminare (es. per cambio di modalità) mentre una registrazione
    # stava per essere arrestata, esegui un ultimo tentativo di inviare il segnale di stop.
    if is_shutting_down:
        try:
            output_queue.put(None, timeout=0.5)
            logging.info("[PROCESS] Segnale di stop finale inviato prima della terminazione.")
        except queue.Full:
            logging.error("[PROCESS] Impossibile inviare il segnale di stop finale. Il video potrebbe non essere finalizzato.")

def writer_thread(state_event):
    """
    THREAD CRITICO: Scrittore dei file video per METEOR_FINDER.
    - Consumatore della `output_queue`.
    - Isola la lenta operazione di I/O su disco.
    - Ascolta un segnale `None` (sentinella) per chiudere e finalizzare correttamente
      il file video, prevenendo la corruzione dei dati.
    """
    # Gestisce la scrittura dei frame video su disco.
    global out, ffmpeg_proc # Still needs access to the global file handles

    # The main loop checks for both the global running flag and the mode-specific event
    while running and state_event.is_set():
        try:
            # Attende un oggetto dalla coda. Potrebbe essere un frame o il segnale di stop (None).
            item = output_queue.get_nowait()

            # --- Logica di Chiusura Coordinata ---
            # Controlla se l'oggetto ricevuto è il segnale di "Fine Stream".
            if item is None:
                logging.info("[WRITE] Segnale di stop ricevuto. Finalizzazione del video in corso...")
                
                # Chiude in modo sicuro il processo ffmpeg, se attivo
                if ffmpeg_proc:
                    if ffmpeg_proc.stdin:
                        try:
                            ffmpeg_proc.stdin.close()
                        except BrokenPipeError:
                            # Questo può accadere se il processo è terminato in anticipo, è sicuro ignorarlo.
                            logging.warning("[WRITE] stdin di ffmpeg già chiuso.")
                            pass
                    ffmpeg_proc.wait() # Attende che il processo ffmpeg termini completamente
                    ffmpeg_proc = None # Resetta la variabile globale
                
                # Chiude in modo sicuro l'oggetto VideoWriter di OpenCV, se attivo
                if out:
                    out.release()
                    out = None # Resetta la variabile globale
                
                logging.info("[WRITE] Video finalizzato e chiuso correttamente.")
                
                # Continua al prossimo ciclo per attendere un nuovo task (non esce dal loop)
                continue

            # --- Logica di Scrittura del Frame ---
            # Se l'oggetto non è None, allora è un frame da scrivere.
            # Scriviamo solo se uno dei writer è effettivamente attivo.
            if out or ffmpeg_proc:
                try:
                    if ffmpeg_proc:
                        ffmpeg_proc.stdin.write(item.tobytes())
                    elif out:
                        # Converte al volo in grayscale per i codec non-H264
#                        frame_gray = item[:height, :width]
                        out.write(item)
                    update_perf_counter("written") # Se vuoi ripristinare il contatore
                except Exception as e:
                    # Gestisce errori che potrebbero verificarsi durante la scrittura
                    logging.error(f"[WRITE] Errore imprevisto durante la scrittura del frame: {e}")
                    # In caso di errore grave, è meglio resettare lo stato di registrazione
                    # per evitare un ciclo di errori.
                    if ffmpeg_proc and ffmpeg_proc.stdin:
                        try:
                            ffmpeg_proc.stdin.close()
                        except BrokenPipeError:
                            pass
                        ffmpeg_proc.wait()
                        ffmpeg_proc = None
                        if out:
                            out.release()
                            out = None
                    recording_event.clear()
        except queue.Empty:
            # È normale che la coda sia vuota, continua semplicemente ad attendere.
            time.sleep(0.01)
            continue
        
    logging.info("[WRITER] Thread di scrittura terminato.")

def snapshot_writer_thread(state_event):
    """Scrittore dei file snapshot per METEOR_FINDER. Deve controllare `state_event` per terminare correttamente."""
    while running and state_event.is_set():
        try:
            snapshot_path, frame_to_save = snapshot_queue.get_nowait()
            start = time.time()
            cv2.imwrite(snapshot_path, frame_to_save)
            duration = time.time() - start
            logging.info(f"[SNAPSHOT] Snapshot salvato in {snapshot_path} ({duration:.3f}s)")
        except queue.Empty:
            time.sleep(0.01)
            continue
        except Exception as e:
            logging.error(f"[SNAPSHOT] Errore nel salvataggio snapshot: {e}")

# --- Timelapse Threads ---
def timelapse_capture_thread(state_event):
    """
    Cattura scatti a lunga esposizione. Usa il pattern `capture_request`/`wait`
    con un timeout personalizzato per gestire in modo robusto esposizioni lunghe.
    """
    logging.info("[TIMELAPSE] Thread di cattura avviato.")
#    last_capture_time = time.time() - APP_CONFIG['timelapse_interval'] # Per scattare subito la prima foto

    # Determina la modalità di cattura
    is_manual_exposure = APP_CONFIG['timelapse_exposure'].lower() != "automatic"
    manual_exposure_time = 0
    if is_manual_exposure:
        try:
            manual_exposure_time = int(APP_CONFIG['timelapse_exposure'])
        except ValueError:
            logging.error(f"[TIMELAPSE] Valore di esposizione manuale non valido: '{APP_CONFIG['timelapse_exposure']}'. Arresto del thread.")
            return    
    
    while running and state_event.is_set():
#        if time.time() - last_capture_time >= args.timelapse_interval:
#            last_capture_time = time.time()
        request = None # Inizializza per il blocco finally
        try:
            if is_manual_exposure:
                 # --- Flusso per Esposizione Lunga e Manuale ---
                logging.info(f"[TIMELAPSE] Avvio cattura manuale (esposizione: {manual_exposure_time}s)...")
                job = picam2.capture_request(wait=False)
                wait_timeout = manual_exposure_time + 5
                request = picam2.wait(job, timeout=wait_timeout * 1000)
                if request is None: raise RuntimeError("La richiesta di cattura manuale è scaduta.")
            else:
                # --- Flusso per Esposizione Automatica ---
                logging.info(f"[TIMELAPSE] Avvio cattura automatica...")
                request = picam2.capture_request() # Semplice chiamata bloccante
                
            captured_image = request.make_array('main')
                
            if APP_CONFIG['timelapse_color']:
                final_frame = cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR)
            else:
                final_frame = cv2.cvtColor(captured_image, cv2.COLOR_YUV2GRAY_I420)
                
            timelapse_writer_queue.put(final_frame)
            logging.info("[TIMELAPSE] Immagine catturata e inviata per il salvataggio.")

        except Exception as e:
            logging.error(f"[TIMELAPSE] Errore durante la cattura: {e}")
        finally:
            if request:
                request.release()

        # Dopo che tutto è finito (sia in caso di successo che di errore), 
        # attendi per l'intervallo specificato dall'utente.
        logging.info(f"[TIMELAPSE] Inizio intervallo di attesa di {APP_CONFIG['timelapse_interval']} secondi...")
        # Usiamo un ciclo per controllare l'evento di stop più frequentemente (ogni secondo).
        for _ in range(APP_CONFIG['timelapse_interval']):
            if not (running and state_event.is_set()):
                break # Interrompi l'attesa se lo script si sta fermando.
            time.sleep(1)        

    logging.info("[TIMELAPSE] Thread di cattura terminato.")

def timelapse_writer_thread(state_event):
    """Salva su disco le immagini finali del timelapse."""
    timelapse_dir = os.path.join(APP_CONFIG['output_dir'], "timelapse")
    os.makedirs(timelapse_dir, exist_ok=True)
    logging.info(f"[TIMELAPSE] Thread di scrittura avviato. Salvataggio in: {timelapse_dir}")
    
    while running and state_event.is_set():
        try:
            frame = timelapse_writer_queue.get_nowait()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(timelapse_dir, f"tl_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            update_perf_counter("events")
        except queue.Empty:
            time.sleep(0.01)
            continue
    logging.info("[TIMELAPSE] Thread di scrittura terminato.")

def run_application():
    """
    Questa è la funzione principale che avvia la state machine e tutti i thread.
    Viene chiamata solo dopo che la configurazione è stata finalizzata.
    """
    global running, current_state, out, ffmpeg_proc, recording_event, reference_frame
    global last_event_time, record_start_time
    global frames_captured, frames_processed, frames_written, events_triggered, lock_perf

    # === LOGGING SETUP (Now safe to call) ===
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if APP_CONFIG['no_log_events']: logger.setLevel(logging.CRITICAL + 1)
    for handler in logger.handlers[:]: logger.removeHandler(handler)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler("detection_log.log")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    console_formatter = ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # === PARAMETER SETUP ===
    STRATEGY = APP_CONFIG['strategy']
    RESOLUTIONS = {"small": (640, 480), "medium": (1280, 720), "large": (1920, 1080)}
    METEOR_EXPOSURE_LIMITS = (50000, 1000000)        

    # Code di comunicazione thread-safe tra i vari componenti.
    frame_queue = queue.Queue(maxsize=APP_CONFIG['frame_queue_maxsize'])
    output_queue = queue.Queue(maxsize=APP_CONFIG['output_queue_maxsize'])
    snapshot_queue = queue.Queue(maxsize=APP_CONFIG['snapshot_queue_maxsize'])
    timelapse_writer_queue = queue.Queue(maxsize=10)

    # === CAMERA SETUP ===
    os.makedirs(APP_CONFIG['output_dir'], exist_ok=True)
    picam2 = Picamera2()
    HAS_AUTOFOCUS = is_autofocus_camera(picam2)
    width, height = RESOLUTIONS[APP_CONFIG['size']]
    
    # === CODEC ===
    # Determina l'estensione del file e il FourCC per i codec non-H264.
    if APP_CONFIG['codec'] == "avi":
        # fourcc = cv2.VideoWriter_fourcc(*'IYUV') # Old
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # More compatible
        extension = "avi"
    elif APP_CONFIG['codec'] == "mjpeg":
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        extension = "avi"
    elif APP_CONFIG['codec'] == "h264":
        fourcc = None
        extension = "mp4"
    else:
        raise ValueError("Codec non supportato.")    
    
    # === MAIN STATE MACHINE ===
    active_threads = []
    state_events = {'meteor_finder': threading.Event(), 'timelapse': threading.Event()}
 
    # Initialize global variables for threads
    reference_frame = None
    recording_event = threading.Event()
    out, ffmpeg_proc = None, None
    running = True
    current_state = "IDLE"
    last_event_time, record_start_time = 0, 0
    frames_captured, frames_processed, frames_written, events_triggered = 0, 0, 0, 0
    lock_perf = threading.Lock()
    
    monitor_t = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
    monitor_t.start()
 
    shutdown_time_obj = None
    if APP_CONFIG['shutdown_time']:
        try:
            shutdown_time_obj = datetime.strptime(APP_CONFIG['shutdown_time'], "%H:%M").time()
            logging.info(f"[MANAGER] Spegnimento del sistema programmato per le {APP_CONFIG['shutdown_time']}")
        except (ValueError, TypeError):
            logging.error(f"[MANAGER] Formato orario di spegnimento non valido.")
    
    shutdown_initiated = False
    shutdown_for_system = False
    
    try:
        while running:
            desired_state = "IDLE"
            is_meteor_time = is_time_in_interval(APP_CONFIG['meteor_start_time'], APP_CONFIG['meteor_end_time'])
            is_timelapse_time = is_time_in_interval(APP_CONFIG['timelapse_start_time'], APP_CONFIG['timelapse_end_time'])
            if is_meteor_time:
                desired_state = "METEOR_FINDER"
            elif is_timelapse_time:
                desired_state = "TIMELAPSE"
            elif current_state == "TIMELAPSE" and not is_timelapse_time and APP_CONFIG['timelapse_to_video']:
                desired_state = "POST_PROCESSING"

            if desired_state != current_state:
                logging.info(f"[MANAGER] Transizione di stato: {current_state} -> {desired_state}", extra={'state': desired_state})
                if current_state != "IDLE":
                    logging.info(f"[MANAGER] Arresto della modalità {current_state}...")
                    state_events[current_state.lower()].clear()
                    for t in active_threads: t.join(timeout=15)
                    alive_threads = [t for t in active_threads if t.is_alive()]
                    if alive_threads:
                        logging.warning(f"[MANAGER] I seguenti thread non sono terminati: {[t.name for t in alive_threads]}")
                    active_threads.clear()
                    if picam2.started:
                        picam2.stop()
                    logging.info(f"[MANAGER] Modalità {current_state} arrestata.")
                
                current_state = desired_state
                if current_state == "METEOR_FINDER":
                    logging.info("[MANAGER] Riconfigurazione camera per METEOR_FINDER...")
                    lores_w = width // APP_CONFIG['downscale_factor']
                    lores_h = height // APP_CONFIG['downscale_factor']
                    meteor_controls = {"FrameDurationLimits": METEOR_EXPOSURE_LIMITS, "AnalogueGain": APP_CONFIG['gain']}
                    if APP_CONFIG['framerate_mode'] == 'fixed':
                        meteor_controls["FrameRate"] = APP_CONFIG['framerate']
                    video_config = picam2.create_video_configuration(
                        main={"size": (width, height), "format": "YUV420"},
                        lores={"size": (lores_w, lores_h), "format": "YUV420"},
                        controls=meteor_controls
                    )
                    picam2.configure(video_config)
                    picam2.start()
                    effective_framerate = APP_CONFIG['framerate']
                    if APP_CONFIG['framerate_mode'] == 'dynamic':
                        metadata = picam2.capture_metadata()
                        effective_framerate = metadata.get("FrameRate", APP_CONFIG['framerate'])
                    logging.info(f"[CAMERA] Framerate effettivo impostato a: {effective_framerate:.2f} fps")
                    pre_event_buffer = deque(maxlen=int(APP_CONFIG['pre_event_seconds'] * effective_framerate))
                    active_threads = [
                        threading.Thread(target=capture_thread_meteor, args=(state_events['meteor_finder'], width, height), daemon=True, name="MeteorCapture"),
                        threading.Thread(target=processing_thread, args=(state_events['meteor_finder'], effective_framerate, pre_event_buffer, width, height), daemon=True, name="MeteorProcess"),
                        threading.Thread(target=writer_thread, args=(state_events['meteor_finder'],), daemon=True, name="MeteorWriter"),
                        threading.Thread(target=snapshot_writer_thread, args=(state_events['meteor_finder'],), daemon=True, name="MeteorSnapshot")
                    ]
                    state_events['meteor_finder'].set()
                    for t in active_threads: t.start()
                
                elif current_state == "TIMELAPSE":
                    logging.info("[MANAGER] Riconfigurazione camera per TIMELAPSE...")
                    timelapse_controls = {}
                    exposure_val = str(APP_CONFIG['timelapse_exposure']).lower()
                    if exposure_val == "automatic":
                        logging.info("[MANAGER] Modalità Timelapse: ESPOSIZIONE AUTOMATICA")
                        timelapse_controls = {"AeEnable": True, "AwbEnable": True}
                        capture_format = "RGB888" if APP_CONFIG['timelapse_color'] else "YUV420"
                    else:
                        logging.info(f"[MANAGER] Modalità Timelapse: ESPOSIZIONE MANUALE ({exposure_val}s)")
                        try:
                            exposure_time_us = int(exposure_val) * 1000000
                            timelapse_controls = {"AeEnable": False, "AwbEnable": False, "AnalogueGain": APP_CONFIG['timelapse_gain'], "ExposureTime": exposure_time_us}
                            if HAS_AUTOFOCUS:
                                timelapse_controls["AfMode"] = controls.AfModeEnum.Manual
                                timelapse_controls["LensPosition"] = 0.0
                            capture_format = "RGB888" if APP_CONFIG['timelapse_color'] else "YUV420"
                        except (ValueError, TypeError):
                            logging.error(f"[MANAGER] Valore di esposizione non valido: '{exposure_val}'. Impossibile avviare.")
                            current_state = "IDLE"
                            continue
                    still_config = picam2.create_still_configuration(main={"size": (width, height), "format": capture_format}, controls=timelapse_controls)
                    picam2.configure(still_config)
                    picam2.start()
                    active_threads = [
                        threading.Thread(target=timelapse_capture_thread, args=(state_events['timelapse'], width, height), daemon=True, name="TimelapseCapture"),
                        threading.Thread(target=timelapse_writer_thread, args=(state_events['timelapse'],), daemon=True, name="TimelapseWriter")
                    ]
                    state_events['timelapse'].set()
                    for t in active_threads: t.start()
                
                elif current_state == "POST_PROCESSING":
                    logging.info("[MANAGER] Ingresso in modalità POST_PROCESSING...", extra={'state': 'POST_PROCESSING'})
                    timelapse_dir = os.path.join(APP_CONFIG['output_dir'], "timelapse")
                    video_filename = os.path.join(APP_CONFIG['output_dir'], f"timelapse_{datetime.now().strftime('%Y%m%d')}.mp4")
                    create_timelapse_video(image_folder=timelapse_dir, output_filename=video_filename, fps=APP_CONFIG['timelapse_video_fps'], cleanup=APP_CONFIG['timelapse_cleanup_images'])
                    current_state = "IDLE"
                    logging.info("[MANAGER] Post-processing completato. Ritorno in modalità IDLE.", extra={'state': 'IDLE'})
                
                elif current_state == "IDLE":
                    logging.info("[MANAGER] Ingresso in modalità IDLE.", extra={'state': 'IDLE'})

            if shutdown_time_obj and not shutdown_initiated:
                if datetime.now().time() >= shutdown_time_obj:
                    logging.info(f"[MANAGER] Orario di spegnimento raggiunto. Avvio della terminazione.", extra={'state': 'SHUTDOWN'})
                    shutdown_initiated = True
                    shutdown_for_system = True
                    running = False
            
            time.sleep(30)

    except KeyboardInterrupt:
        logging.info("[MAIN] Terminazione richiesta dall'utente...")
    finally:
        running = False
        for event in state_events.values(): event.clear()
        if 'recording_event' in globals() and recording_event.is_set():
            logging.info("[MAIN] Finalizzazione della registrazione in corso...")
            output_queue.put(None)
        all_threads = [monitor_t] + active_threads
        for t in all_threads: t.join(timeout=5)
        if picam2.started: picam2.stop()
        if out: out.release()
        if ffmpeg_proc:
            if ffmpeg_proc.stdin:
                try: ffmpeg_proc.stdin.close()
                except BrokenPipeError: pass
            ffmpeg_proc.wait()
        logging.info("[MAIN] Uscita completata.")

    if shutdown_for_system:
        logging.info("[SHUTDOWN] Esecuzione del comando di spegnimento del sistema...", extra={'state': 'SHUTDOWN'})
        # ... (shutdown messages)
        os.sync() 
        time.sleep(2)
        os.system("sudo shutdown now")


# === MAIN: The State Machine Manager ===
if __name__ == "__main__":
    """
    SEZIONE CRITICA: Il ciclo di controllo principale dell'applicazione.
    - Agisce come uno scheduler, controllando l'ora per determinare lo stato desiderato.
    - Gestisce il ciclo di vita di tutti i thread (avvio, arresto, join).
    - Esegue le transizioni di stato in modo sicuro, arrestando i vecchi thread
      e riconfigurando la camera prima di avviare i nuovi.
    - Gestisce lo spegnimento programmato e la chiusura pulita (`KeyboardInterrupt`).
    """
    
    # 1. Definisci i valori predefiniti di base.
    master_defaults = {
        "size": "medium", "binning": True, "gain": 8.0, "codec": "h264", "output_dir": "output",
        "framerate_mode": "fixed", "framerate": 30 if pi_model == "pi5" else 15,
        "meteor_start_time": "22:00", "meteor_end_time": "05:00",
        "strategy": "contour", "record_duration": 12, "pre_event_seconds": 5,
        "min_brightness": 50, "min_area": 5,
        "diff_threshold": 25, "trigger_area": 4000, "learning_rate": 0.01,
        "timelapse_start_time": None, "timelapse_end_time": None,
        "timelapse_exposure": "30", "timelapse_interval": 35,
        "timelapse_gain": 8.0, "timelapse_color": False,
        "timelapse_to_video": False, "timelapse_video_fps": 24, "timelapse_cleanup_images": False,
        "no_log_events": False, "shutdown_time": None,
        "frame_queue_maxsize": 20, "output_queue_maxsize": 120,
        "snapshot_queue_maxsize": 10, "downscale_factor": 4
    }


    # 2. Carica la configurazione da file o crea un nuovo file.
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            try:
                config_data = json.load(f)
            except Exception as e:
                print(f"{Fore.RED}[WARNING] Impossibile analizzare {CONFIG_FILE}: {e}. Verranno usati i predefiniti.{Style.RESET_ALL}")
                config_data = master_defaults
    else:
        print(f"[CONFIG] File di configurazione non trovato. Creazione di {CONFIG_FILE} con i valori predefiniti.")
        save_config(master_defaults)
        config_data = master_defaults
        

    # 3. Avvia l'editor interattivo.
    final_config = edit_config_interactive(config_data)

    # 4. Se l'utente ha scelto di avviare, popola la configurazione globale e avvia l'applicazione.
    if final_config:
        APP_CONFIG = final_config
        # La funzione run_application() ora contiene l'intera logica della state machine
        # e leggerà da APP_CONFIG.
        
        # We need to re-implement all the rest of the script inside run_application now.
        # This is a massive refactor. Let's assume the user can copy-paste the logic.
        print("\n" + "="*15 + " AVVIO APPLICAZIONE " + "="*15)
        run_application() # This is the conceptual call.
        
    active_threads = []
    state_events = {'meteor_finder': threading.Event(), 'timelapse': threading.Event()}
    
    width, height = RESOLUTIONS[APP_CONFIG['size']]
    
    # --- Shutdown Time Setup ---
    shutdown_time_obj = None
    if APP_CONFIG['shutdown_time']:
        try:
            shutdown_time_obj = datetime.strptime(APP_CONFIG['shutdown_time'], "%H:%M").time()
            logging.info(f"[MANAGER] Spegnimento del sistema programmato per le {APP_CONFIG['shutdown_time']}")
        except ValueError:
            logging.error(f"[MANAGER] Formato dell'orario di spegnimento non valido: {APP_CONFIG['shutdown_time']}. La funzione è disabilitata.")
    
    shutdown_initiated = False
    shutdown_for_system = False

    # Avvia i thread universali
    logging.info("[MAIN] Avvio dei thread universali (Monitor)...")
    monitor_t = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
    monitor_t.start()
    
    try:
        while running:
            # 1. Determina lo stato desiderato in base alla pianificazione
            desired_state = "IDLE"
            is_meteor_time = is_time_in_interval(APP_CONFIG['meteor_start_time'], APP_CONFIG['meteor_end_time'])
            is_timelapse_time = is_time_in_interval(APP_CONFIG['timelapse_start_time'], APP_CONFIG['timelapse_end_time'])

            if is_meteor_time:
                desired_state = "METEOR_FINDER"
            elif is_timelapse_time:
                desired_state = "TIMELAPSE"
            elif current_state == "TIMELAPSE" and not is_timelapse_time and APP_CONFIG['timelapse_to_video']:
                desired_state = "POST_PROCESSING"

            # 2. Se lo stato deve cambiare, esegui la transizione
            if desired_state != current_state:
                logging.info(f"[MANAGER] Transizione di stato: {current_state} -> {desired_state}")

                # --- Fase di SHUTDOWN ---
                if current_state != "IDLE":
                    logging.info(f"[MANAGER] Arresto della modalità {current_state}...")

                    # 1. Signal all threads in the current mode to stop their loops.
                    state_events[current_state.lower()].clear()
                    
                    # 2. Wait for them to finish gracefully.
                    #    The writer thread will finish its current file thanks to the sentinel.
                    for t in active_threads:
                        t.join(timeout=10) # Give them a generous timeout to finish I/O

                    # 3. Check for any that did not terminate.
                    alive_threads = [t for t in active_threads if t.is_alive()]
                    if alive_threads:
                        logging.warning(f"[MANAGER] I seguenti thread non sono terminati: {[t.name for t in alive_threads]}")
                    
                    active_threads.clear()
                    
                    # 4. Only stop the camera after all threads that use it are confirmed down.
                    if picam2.started:                        
                        picam2.stop()
                        
                    logging.info(f"[MANAGER] Modalità {current_state} arrestata.")
                
                # --- Fase di STARTUP ---
                current_state = desired_state
                if current_state == "METEOR_FINDER":
                    # ... (Logica di configurazione e avvio thread per METEOR_FINDER)
                    logging.info("[MANAGER] Riconfigurazione della camera per METEOR_FINDER...")
                    lores_w = width // APP_CONFIG['downscale_factor']
                    lores_h = height // APP_CONFIG['downscale_factor']
                    
                    # Costruisce i controlli di base
                    meteor_controls = {"FrameDurationLimits": METEOR_EXPOSURE_LIMITS, "AnalogueGain": APP_CONFIG['gain']}
                    
                    # Aggiunge il controllo del framerate solo se la modalità è 'fixed'
                    if APP_CONFIG['framerate_mode'] == 'fixed':
                        meteor_controls["FrameRate"] = APP_CONFIG['framerate']
                        
                    video_config = picam2.create_video_configuration(
                        main={"size": (width, height), "format": "YUV420"},
                        lores={"size": (lores_w, lores_h), "format": "YUV420"},
                        controls=meteor_controls
                    )
                    picam2.configure(video_config)
                    picam2.start()

                    if APP_CONFIG['framerate_mode'] == 'dynamic':
                        metadata = picam2.capture_metadata()
                        effective_framerate = metadata.get("FrameRate")
                        logging.info(f"[CAMERA] Framerate dinamico rilevato: {effective_framerate:.2f} fps")
                    else: # 'fixed'
                        effective_framerate = APP_CONFIG['framerate']
                        logging.info(f"[CAMERA] Framerate fisso impostato a: {effective_framerate} fps")

                    logging.info(f"[MANAGER] Creazione del pre-event buffer per {APP_CONFIG['pre_event_seconds']} secondi ({int(APP_CONFIG['pre_event_seconds'] * effective_framerate)} frames).")                   
                    pre_event_buffer = deque(maxlen=int(APP_CONFIG['pre_event_seconds'] * effective_framerate))

                    active_threads = [
                        threading.Thread(target=capture_thread_meteor, args=(state_events['meteor_finder'], width, height), daemon=True, name="MeteorCapture"),
                        threading.Thread(target=processing_thread, args=(state_events['meteor_finder'], effective_framerate, pre_event_buffer, width, height), daemon=True, name="MeteorProcess"),
                        threading.Thread(target=writer_thread, args=(state_events['meteor_finder'],), daemon=True, name="MeteorWriter"),
                        threading.Thread(target=snapshot_writer_thread, args=(state_events['meteor_finder'],), daemon=True, name="MeteorSnapshot")                    
                    ]
                    state_events['meteor_finder'].set()
                    for t in active_threads: t.start()
                
                elif current_state == "TIMELAPSE":
                    # ... (Logica di configurazione e avvio thread per TIMELAPSE)
                    logging.info("[MANAGER] Riconfigurazione della camera per TIMELAPSE...")
 
                    timelapse_controls = {}
                    exposure_val = APP_CONFIG['timelapse_exposure'].lower() 
 
                    if exposure_val == "automatic":
                        # Modalità Automatica: lascia che la camera decida tutto
                        logging.info("[MANAGER] Modalità Timelapse: ESPOSIZIONE AUTOMATICA")
                        timelapse_controls = {
                            "AeEnable": True,  # Abilita Auto Exposure
                            "AwbEnable": True, # Abilita Auto White Balance
                        }
                        capture_format = "RGB888" if APP_CONFIG['timelapse_color'] else "YUV420"
                    else:
                        # Modalità Manuale: imposta valori fissi
                        logging.info(f"[MANAGER] Modalità Timelapse: ESPOSIZIONE MANUALE ({exposure_val}s)")
                        try:
                            exposure_time_us = int(exposure_val) * 1000000
                            timelapse_controls = {
                                "AeEnable": False, "AwbEnable": False,
                                "AnalogueGain": APP_CONFIG['timelapse_gain'],
                                "ExposureTime": exposure_time_us,
                            } 
                            if HAS_AUTOFOCUS:
                                timelapse_controls["AfMode"] = controls.AfModeEnum.Manual
                                timelapse_controls["LensPosition"] = 0.0
                            capture_format = "RGB888" if APP_CONFIG['timelapse_color'] else "YUV420"
                        except ValueError:
                            logging.error(f"[MANAGER] Valore di esposizione non valido: '{exposure_val}'. Impossibile avviare il timelapse.")
                            current_state = "IDLE" # Torna a IDLE in caso di errore
                            continue
                            
                    still_config = picam2.create_still_configuration(main={"size": (width, height), "format": capture_format}, controls=timelapse_controls)
                    picam2.configure(still_config)
                    picam2.start()                            

                    active_threads = [
                        threading.Thread(target=timelapse_capture_thread, args=(state_events['timelapse'],), daemon=True, name="TimelapseCapture"),
                        threading.Thread(target=timelapse_writer_thread, args=(state_events['timelapse'],), daemon=True, name="TimelapseWriter")
                    ]
                    state_events['timelapse'].set()
                    for t in active_threads: t.start()
                
                elif current_state == "POST_PROCESSING":
                    # ... (Logica per avviare la creazione del video)                    
                    logging.info("[MANAGER] Ingresso in modalità POST_PROCESSING per la creazione del video timelapse.")
                    timelapse_dir = os.path.join(APP_CONFIG['output_dir'], "timelapse")
                    video_filename = os.path.join(APP_CONFIG['output_dir'], f"timelapse_{datetime.now().strftime('%Y%m%d')}.mp4")
                    
                    # Esegui la funzione di creazione video
                    create_timelapse_video(
                        image_folder=timelapse_dir,
                        output_filename=video_filename,
                        fps=APP_CONFIG['timelapse_video_fps'],
                        cleanup=APP_CONFIG['timelapse_cleanup_images']
                    )                
                    # Dopo il post-processing, lo stato naturale successivo è IDLE
                    current_state = "IDLE" 
                    logging.info("[MANAGER] Post-processing completato. Ritorno in modalità IDLE.")               
                
                elif current_state == "IDLE":
                    logging.info("[MANAGER] Ingresso in modalità IDLE.")

            # 3. --- Controllo Spegnimento Programmato ---
            if shutdown_time_obj and not shutdown_initiated:
                if datetime.now().time() >= shutdown_time_obj:
                    logging.info(f"[MANAGER] Orario di spegnimento ({APP_CONFIG['shutdown_time']}) raggiunto. Avvio della terminazione.", extra={'state': 'SHUTDOWN'})
                    shutdown_initiated = True  # Assicura che questo blocco venga eseguito una sola volta
                    shutdown_for_system = True # Flag per eseguire il comando di spegnimento dopo la pulizia
                    running = False            # Avvia la terminazione pulita dello script             
            
            time.sleep(30) # Intervallo di controllo dello scheduler

    except KeyboardInterrupt:
        logging.info("[MAIN] Terminazione richiesta dall'utente...")
    finally:
        running = False
        for event in state_events.values(): event.clear()
        
        # Ensure any final video is saved
        if 'recording_event' in globals() and recording_event.is_set():
            logging.info("[MAIN] Finalizzazione della registrazione in corso...")
            output_queue.put(None)

        all_threads = [monitor_t] + active_threads
        for t in all_threads: t.join(timeout=5)
        
        if picam2.started: picam2.stop()
        
        # This final check is belt-and-suspenders, but safe
        if out: out.release()
        if ffmpeg_proc:
            if ffmpeg_proc.stdin:
                try: ffmpeg_proc.stdin.close()
                except BrokenPipeError: pass
            ffmpeg_proc.wait()
        
        logging.info("[MAIN] Uscita completata.")

    # --- Fase di Spegnimento del Sistema ---
    # Questo blocco viene eseguito solo dopo che il blocco `finally` è completato
    if shutdown_for_system:
        logging.info("[SHUTDOWN] Esecuzione del comando di spegnimento del sistema (sudo shutdown now)...", extra={'state': 'SHUTDOWN'})
        logging.info("**************************************************", extra={'state': 'SHUTDOWN'})
        logging.info(f"[SHUTDOWN] Spegnimento del sistema programmato per le {APP_CONFIG['shutdown_time']} in corso...", extra={'state': 'SHUTDOWN'})
        logging.info("**************************************************", extra={'state': 'SHUTDOWN'})
        # Svuota i buffer del sistema operativo per assicurarsi che i log siano scritti su disco
        os.sync() 
        time.sleep(2) # Breve attesa per sicurezza
        
        # Esegui il comando di spegnimento
        # NOTA: Lo script deve essere eseguito con `sudo` affinché questo comando funzioni.
        os.system("sudo shutdown now")
