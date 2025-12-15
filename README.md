# Unitree G1 Simulation mit NVIDIA Isaac Lab

Dieses Repository enthaelt Workflows und Skripte, um den **Unitree G1** in **NVIDIA Isaac Sim** und **Isaac Lab** zu simulieren. Es deckt sowohl hartcodierte kinematische Bewegungen als auch Deep-Reinforcement-Learning-Setups für Lokomotion ab.

## 📋 Projektüberblick

Ziel ist ein Simulations-Setup, in dem der G1 autonome Gehbewegungen per Reinforcement Learning (PPO) erlernt. Dabei wird der gesamte Sim-to-Real-Weg von URDF-Import bis Trainingskonfiguration adressiert.

**Highlights:**
* **URDF → USD:** Import und Aufsetzen des G1-Modells für Isaac Sim (inkl. Fixes für root link).
* **Kinematische Kontrolle:** Vordefinierte Bewegungen zum schnellen Check der Gelenke (Winken, Tanzen).
* **RL-Umgebung:** Eigener Wrapper und Konfiguration für `RSL_RL` Training.
* **Domain Randomization:** Variation von Masse, Reibung und Stoesseln für robustes Training.

## 🔧 Aktuelle Verbesserungen

- **Stabiler Boden:** Feste Bodenebene wird in den Workflows gespawnt (hohe Reibung, `restitution=0.0`), damit der Roboter nicht „spickt“.
- **Robustere Physik:** Kleinere Zeitschritte (`dt=1/120`), erhoehte Solver-Iterationen, aktivierte Stabilization und `bounce_threshold_velocity=0.0` fuer ruhigeres Kontaktverhalten.
- **Fixierbare Basis:** Neue CLI-Option `--free-root` in `run_g1_move.py`, `run_g1_dance.py` und `run_g1_walk.py`. Standard: Basis fixiert (stabil). Mit `--free-root` ist die Basis frei (kann umfallen).
- **Trained Playback:** Neues Skript `trained_g1_walk.py` zum Abspielen eines trainierten TorchScript-Modells (z. B. `motion.pt`). Empfohlen mit `--num_envs 1` (LSTM/TorchScript). Aktionen werden bei Bedarf auf die erwartete Dimensionszahl der Umgebung reduziert.
- **Dependency-Fix:** Kit-Experience `apps/isaaclab.python.kit` nutzt jetzt die lokal verfügbare URDF-Importer-Version `2.4.19` (statt `2.4.31`), um den Resolver-Fehler zu vermeiden.

## ⚙️ Installation & Setup

1. **Voraussetzungen**
     * NVIDIA Isaac Sim (4.x oder 5.x)
     * Isaac Lab installiert
     * `RSL_RL` Bibliothek für Reinforcement Learning

2. **Repository klonen**
     ```bash
     git clone https://github.com/thinih51/IsaacLab.git
     cd IsaacLab
     ```

3. **Pfad konfigurieren**
     In den Python-Skripten sicherstellen, dass `USD_PATH` auf deine lokale USD zeigt:
     ```python
     USD_PATH = "C:/Users/tshed/Documents/WIP/IsaacLab/resources/g1/g1_29dof/g1_29dof.usd"
     ```

## 🚀 Nutzung

Skripte mit dem Isaac-Lab-Launcher starten (z. B. `./isaaclab.bat -p` unter Windows oder `./isaaclab.sh -p` unter Linux).

### 1. Hartcodierte Bewegungen
Schneller Gelenk-Check ohne Neuronale Netze.

* **Arm Waving (Basis-Test):**
    ```bash
    ./isaaclab.bat -p source/standalone/workflows/run_g1_move.py
    ```
* **Dance Mode (Koordinationstest):**
    ```bash
    ./isaaclab.bat -p source/standalone/workflows/run_g1_dance.py
    ```

Optional: Basis freigeben (kann umfallen)

    ```bash
    ./isaaclab.bat -p source/standalone/workflows/run_g1_dance.py --free-root
    ```

### 2. Reinforcement Learning (Training)
PPO-Training zum Laufen:
* **Kommando:**
    ```bash
    ./isaaclab.bat -p source/standalone/workflows/run_g1_train.py --num_envs 4096 --headless
    ```
* **Setup:** **4096 parallele Environments** mit Domain Randomization (Masse, Bodenreibung, externe Pushes), um Overfitting zu vermeiden.

### 3. Inference (Play)
Trainiertes Modell abspielen:
* **Kommando:**
    ```bash
    ./isaaclab.bat -p source/standalone/workflows/run_g1_play.py --num_envs 16
    ```

### 4. Trained Playback (TorchScript)
Eigene trainierte Policy als TorchScript (`motion.pt`) abspielen:

```bash
./isaaclab.bat -p source/standalone/workflows/trained_g1_walk.py --model_path trained_model/motion.pt --num_envs 1
```

Hinweise:
- Empfohlen ist `--num_envs 1`, da viele TorchScript/LSTM-Exports einen internen Zustand mit Batch=1 erwarten.
- Das Skript passt bei Bedarf die Aktionsdimension des Modells an die Umgebung an (z. B. 12 → 6 Gelenke). Fuer exakte Zuordnung kann eine benutzerdefinierte Mapping-Liste hinterlegt werden.
- Falls die Observation-Dimension nicht zum Modell passt, die Observations-Konfiguration im Skript entsprechend anpassen (z. B. 47-Dim Setup).

## 📊 Ergebnisse & Beobachtungen

Mehrere Trainingslaeufe von **30 Minuten bis 6 Stunden**.

### Physik-Check
PhysX simuliert Gravitation, Kollision und Reibung korrekt. Das Fallen bestaetigt, dass der Roboter dynamisch ist (`fix_root_link=False`) und mit der Bodenebene interagiert.

### Trainings-Herausforderungen
Trotz 10.000 Iterationen und Domain Randomization kein stabiles Gangbild innerhalb von 6 Stunden. Das deckt sich mit aktüller Forschung: Humanoide Lokomotion braucht oft **mehrtaegiges Training** auf starker Hardware, bis eine robuste Policy entsteht.

## 📂 Dateiübersicht

* `run_g1_move.py`: Kinematischer Basistest (Armbewegung).
* `run_g1_dance.py`: Kinematische Koordination (Beine, Arme, Hüfte).
* `run_g1_train.py`: PPO-Training mit Domain Randomization und Rewards.
* `run_g1_play.py`: Inference-Skript zum Laden von `model.pt` und Anzeigen im Viewport.
* `trained_g1_walk.py`: Abspielen eines TorchScript-Modells (`motion.pt`) mit Empfehlung `--num_envs 1`.

## 🧩 Troubleshooting

- "Failed to resolve extension dependencies" mit `isaacsim.asset.importer.urdf`:
    - In `apps/isaaclab.python.kit` wurde die Abhaengigkeit auf `"isaacsim.asset.importer.urdf" = {version = "2.4.19", exact = true}` gesetzt, passend zur lokal verfügbaren Version.
    - Danach erneut starten: `./isaaclab.bat -p <skript>`

## 📝 Lizenz

Nur zu Lehr-/Demozwecken.