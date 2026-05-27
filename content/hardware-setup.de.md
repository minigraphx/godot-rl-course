# Hardware-Setup & Trainingszeit

[Kursstartseite](index.md)

---

Wähle deine Hardware mit Bedacht. Dieser Leitfaden hilft dir zu verstehen, was du brauchst, um jede Unit zu trainieren, wie lange es dauert und was du tun kannst, wenn dein Laptop nicht ausreicht.

---

## 1 · Rechenanforderungen pro Unit

Diese Tabelle zeigt, ob die CPU ausreicht, ob eine GPU hilft, und realistische Trainingszeiten je Unit.

| Unit | Aufgabe | CPU ausreichend? | GPU hilfreich? | Geschätzte Zeit (M1 MacBook / RTX 3060 / RTX 4090) |
|------|---------|------------------|----------------|---------------------------------------------------|
| Unit 0–1 | CartPole, Acrobot-Setup | ✅ Ja | ❌ Nein | 30 s / 15 s / 10 s |
| Unit 2 | LunarLander (diskret) | ✅ Ja | ❌ Nein | 3–5 min / 1–2 min / 30 s |
| Unit 3 | CrossTheRoad (DQN, visuell) | ⚠️ Langsam | ✅ Stark empfohlen | 45–90 min / 8–12 min / 2–3 min |
| Unit 4 | JumperHard (PPO, kontinuierlich) | ⚠️ Langsam | ✅ Stark empfohlen | 60–120 min / 10–15 min / 3–5 min |
| Unit 6 | FlyBy (kontinuierliche, visuelle Beobachtungen) | ⚠️ Langsam | ✅ Stark empfohlen | 90–150 min / 12–20 min / 4–6 min |
| Visuelle Beobachtungen (jede Unit) | CNN-Policies + Pixeleingaben | ❌ Nicht praktikabel | ✅ Erforderlich | 2–3 Std / 15–25 min / 5–8 min |
| Capstone-Projekt | 2 Mio. Steps, eigene Aufgabe | ❌ Nicht praktikabel | ✅ Erforderlich | 3–5 Std / 30–45 min / 10–15 min |

!!! info "Annahmen zur Zeit"
    Die Zeiten gehen von 1 Mio. Steps aus, sofern die Unit nichts anderes angibt. Die tatsächliche Dauer hängt ab von:
    - CPU-Taktfrequenz, Kernzahl und RAM-Bandbreite
    - GPU-VRAM-Größe (größere Replay-Buffer brauchen mehr VRAM)
    - Anzahl paralleler Umgebungen (typisch 8–16 beim Training)
    - Hyperparameter-Wahl (Lernrate, Batch-Größe, Entropie-Regularisierung)

---

## 2 · Wann reicht CPU, wann brauchst du eine GPU?

**Schnelle Faustregel:**

| Szenario | CPU OK? | Empfehlung |
|----------|---------|------------|
| MLP-Policies + niedrigdimensionale Beobachtungen (raycasts, ≤16 Dim.) + ≤16 parallele Envs | ✅ Ja | Jede moderne CPU (ab 2018) mit 16 GB RAM |
| CNN-Policies (Pixel-Beobachtungen) | ❌ Nein | GPU erforderlich — CPU-CNN-Training ist 10–50× langsamer |
| Replay-Buffer > 500 k Steps + Pixel-Beobachtungen | ❌ Nein | GPU erforderlich (Speicherbandbreite + Forward-Pass) |
| \>32 parallele Umgebungen | ⚠️ Riskant | GPU oder leistungsstarke Multi-Core-CPU (16+ Kerne) |
| SAC oder anderer kontinuierlicher Algorithmus + visuelle Beobachtungen | ❌ Nein | GPU dringend empfohlen (doppelte Update-Frequenz) |

!!! warning "CPU genügt für Units 0–4"
    Wenn du visuelle Beobachtungen weglässt (stattdessen Raycasts), trainiert eine moderne CPU die Units 0–4 in akzeptabler Zeit. Unit 3 (CrossTheRoad) *kann* mit Raycasts arbeiten; der Kurs enthält beide Varianten.

---

## 3 · Empfohlene Setups nach Budget

| Stufe | Hardware | Kosten | Hinweise |
|-------|----------|--------|----------|
| **Kostenlos** | Google Colab (T4 GPU, 12 Std/Sitzung) | 0 $ | Gut für Units 1–6 und Capstone. Anmelden, Notebook hochladen, los. Das 12-Stunden-Limit erfordert Checkpointing. |
| **Hobby** | Gebrauchte RTX 3060 12 GB (~250 $) oder AMD RX 6700 XT 12 GB (~280 $) | ~250–300 $ | Einstiegsklasse NVIDIA/AMD. Trainiert Units 0–6 in 30–60 min. Gebraucht bei eBay/Marketplace kaufen. Erfordert PCIe-Slot + Netzteil. |
| **Studierende** | RTX 4060 Ti 8 GB (neu, ~450 $) oder RTX 4070 (gebraucht, ~400 $) | ~400–500 $ | 8 GB sind knapp für Pixel-Obs + große Buffer; 12–16 GB ideal. 4070 für zukünftige Projekte empfohlen. |
| **Ernsthaft** | RTX 4070 Ti / 4080 / 4090 (Desktop) | 800–2500 $ | Trainiert Capstone in <15 min. Zukunftssicher für größere Modelle und längeres Training. |
| **Forschung** | Multi-GPU-Workstation (2× RTX 4090 oder A100s in der Cloud) | 5 k $+ lokal oder $/Std in der Cloud | Verteiltes Training, größere Experimente, Rollouts in Forschungsgröße. |

!!! tip "Bestes Preis-Leistungs-Verhältnis für Studierende"
    Eine gebrauchte RTX 3060 oder 4070 bietet die besten Kosten pro Trainingsstunde für den Kurs. Gebrauchte GPUs bei eBay/Marketplace sind oft 50 % günstiger als neu.

---

## 4 · Apple Silicon — was funktioniert und was nicht

**Die Lage bei Apple M1/M2/M3 ist gemischt.** Stable-Baselines3 + PyTorch *laufen* auf Apple Silicon, aber einige Funktionen sind unvollständig.

| Funktion | Status | Workaround |
|----------|--------|-----------|
| MLP-Policies (Units 0–4 ohne visuelle Beobachtungen) | ✅ Funktioniert gut | Standard. Schnelle Inferenz, Training ~50 % langsamer als RTX 3060. |
| Explizites CPU-Training | ✅ Funktioniert | `model = PPO("MlpPolicy", env, device="cpu")` (auch auf GPU-fähigen Maschinen) |
| PyTorch MPS (Metal Performance Shaders) | ⚠️ Unvollständig | Einige SAC- und CNN-Operationen schlagen bei float64 fehl. Gemischte Ergebnisse. |
| CNN-Training (Pixel-Beobachtungen) | ❌ Langsam | 5–10× langsamer als RTX 3060. Funktioniert, aber nicht praktikabel für Units 3, 6, Capstone. |
| Große Replay-Buffer (Pixel-Obs) | ❌ Eng | M1 8 GB/16 GB Unified Memory bremst beim Buffer-I/O. M2/M3 Pro/Max mit 32 GB+ ist besser. |

!!! warning "Verlass dich nicht auf automatische MPS-Erkennung"
    PyTorchs automatischer MPS-Fallback ist fragil. Erzwinge explizit CPU:
    ```python
    model = PPO("MlpPolicy", env, device="cpu")
    ```
    Das Training ist langsamer, aber stabil. Für visuelle Beobachtungen lieber Google Colab nutzen.

!!! tip "M2/M3 Pro/Max mit 32 GB Unified Memory"
    Mit einem M3 Max mit 32 GB kannst du visuelle-Obs-Units trainieren, aber immer noch ~3–5× langsamer als eine RTX 3060. Gut zum Lernen, nicht für Forschung.

---

## 5 · Cloud-Optionen (wenn lokal nicht reicht)

Wenn dein Laptop den Capstone nicht stemmt oder du über mehrere GPUs parallelisieren willst, vermieten Cloud-Anbieter GPUs stundenweise.

| Anbieter | T4 $/Std | A100 $/Std | Einrichtung | Am besten für | Hinweise |
|----------|----------|------------|-------------|---------------|----------|
| **Vast.ai** | 0,10–0,20 | 1,50–2,50 | Mittel | Kostenbewusste Studierende | Spot-Instanzen, große Hardware-Auswahl. Erfordert Docker-Kenntnisse. |
| **Lambda Labs** | 0,50 | 2,50 | Einfach | Hands-off-Training | PyTorch und Jupyter vorinstalliert. Sauberes Linux. Mittlerer Preis, zuverlässig. |
| **Google Colab Pro** | inklusive (T4/A100) | inklusive (selten) | Einfach | Schnelle Experimente | 12 $/Monat, 100 Compute-Units/Monat. Gute UX, limitierte Stunden. |
| **RunPod** | 0,18–0,35 | 2,00–3,00 | Mittel | Studierende in Europa | Gute Uptime, faire Preise, guter Support. |
| **AWS SageMaker** | 0,35 | 4,50 | Schwer | Nur Enterprise | Am teuersten, aber Enterprise-SLAs. Für Studierende nicht empfohlen. |
| **GCP AI Platform** | 0,35 | 4,00 | Schwer | Nur Enterprise | Wie AWS; Enterprise-Preise. |

!!! info "Geschätzte Trainingskosten für den Capstone"
    Capstone-Training (2 Mio. Steps, ~1,5–2 Std auf RTX 3060):
    - Vast.ai: 0,30–0,40 $
    - Lambda Labs: 1,00–1,25 $
    - Colab Pro: inklusive (aber die Sitzung kann ablaufen)
    - RunPod: 0,30–0,70 $
    - AWS: 0,70–0,90 $

!!! tip "Spot-Instanzen sparen 50 %"
    Vast.ai und AWS bieten Spot-Preise. Das Training ist fehlertolerant, wenn du alle 100 k Steps Checkpoints speicherst (SB3 hat eingebautes Checkpointing).

---

## 6 · Festplatte und RAM

**Minimale Systemanforderungen für den Kurs:**

| Ressource | Minimum | Empfohlen | Warum |
|-----------|---------|-----------|-------|
| **RAM** | 16 GB | 32 GB | Replay-Buffer für Pixel-Beobachtungen verbrauchen RAM. Units 0–2 brauchen ~2–4 GB; Units 3+ mit großem Buffer = 12–20 GB. |
| **SSD-Speicher** | 10 GB | 50 GB | Python-Env (~3 GB) + SB3 + Godot (~2 GB). Wenn du alle TensorBoard-Logs und Modell-Checkpoints jeder Unit behältst: weitere 20–50 GB. |
| **Festplattentyp** | HDD (langsam) | SSD (schnell) | Replay-Buffer-I/O ist häufig. SSD reduziert die Trainingszeit für Pixel-Obs um ~20–30 %. |

!!! warning "Festplattenplatz für den Capstone"
    Wenn du 50 Capstone-Varianten mit Checkpoints trainierst, kommst du leicht auf 50 GB. Alte Logs monatlich aufräumen: `rm -rf logs/unit-0*`.

---

## 7 · Schneller Hardware-Check

Führe diesen Python-Schnipsel aus, um zu sehen, welche GPU (falls vorhanden) auf deiner Maschine verfügbar ist:

```python
import torch
import psutil

print("=" * 50)
print("PyTorch GPU Check")
print("=" * 50)

# CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1e9
        print(f"GPU {i}: {props.name}")
        print(f"  VRAM: {total_mem:.1f} GB")

# MPS (Apple Silicon)
if torch.backends.mps.is_available():
    print(f"MPS available: True")
    print("  (Metal Performance Shaders - Apple Silicon)")
else:
    print(f"MPS available: False")

# CPU info
print(f"\nCPU cores: {psutil.cpu_count(logical=False)}")
print(f"CPU (with HT): {psutil.cpu_count(logical=True)}")

# RAM
mem = psutil.virtual_memory()
print(f"RAM: {mem.total / 1e9:.1f} GB total")
print(f"     {mem.available / 1e9:.1f} GB available")

print("=" * 50)
```

Speichere dies als `check_hardware.py` und führe es aus:

```bash
python check_hardware.py
```

!!! info "Worauf achten"
    - **CUDA available: True** → Du hast eine NVIDIA-GPU. Prüfe das VRAM — für visuelle Beobachtungen brauchst du ≥8 GB.
    - **CUDA available: False** + MPS available: True → Du hast Apple Silicon. Für visuelle Aufgaben CPU nutzen.
    - **Alles False** → Du hast nur CPU. Gut für Units 0–4; für Units 3+ Google Colab erwägen.

---

## Nächste Schritte

- **Units 0–4?** Deine aktuelle Maschine reicht (auch CPU-only).
- **Units 3–6 mit visuellen Beobachtungen?** Auf GPU aufrüsten oder Google Colab nutzen.
- **Capstone-Projekt?** GPU (RTX 3060+) oder Cloud-Anbieter, um in <1 Stunde fertig zu werden.

[← Zurück zur Kursstartseite](index.md)
