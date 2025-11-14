# 🤖 Standalone Watus Interface

> **Niezależny monitor emotikonów dla watus_project**  
> Podłącza się automatycznie, nie wymaga modyfikacji plików

## 🚀 30-sekundowy Start

```bash
# 1. Bądź w katalogu watus_project
cd /path/to/your/watus_project_2

# 2. Setup standalone interface
python setup_standalone.py

# 3. Zainstaluj dependencies
cd standalone_interface
pip install -r standalone_requirements.txt

# 4. Uruchom interface (Terminal 1)
python run_standalone.py

# 5. Uruchom watus (Terminal 2)  
python watus.py

# 6. Otwórz w przeglądarce
# http://127.0.0.1:5001
```

## ✨ Co to robi?

- ✅ **Monitoruje** watus_project przez ZMQ i log files
- ✅ **Wyświetla** emotikony w przeglądarce w czasie rzeczywistym  
- ✅ **Auto-sync** ze stanami watus (listening, processing, speaking, etc.)
- ✅ **Zero modyfikacji** - nie zmieniamy watus.py
- ✅ **Plug & Play** - uruchom i działa

## 🎭 Emotikony

| Watus State | Emotikon | Animation |
|-------------|----------|-----------|
| Listening | 🔵 | Pulse |
| Processing | 🟡 | Spin |
| Speaking | 🟢 | Bounce |
| Happy | 🟡 | Shine |
| Error | 🔴 | Shake |
| Standby | 🟢 | Normal |
| Sleep | ⚫ | Fade |

## 🔧 Manual Control

- **Przyciski**: Kliknij aby ręcznie zmienić emotikon
- **Klawiatura**: 1-7 dla różnych stanów
- **API**: http://127.0.0.1:5001/status

## 📁 Struktura Plików

```
watus_project/
├── watus.py                    # Bez zmian!
├── oczyWatusia/                # Bez zmian!
└── standalone_interface/        # NOWY: Niezależny interface
    ├── standalone_watus_interface.py
    ├── start_interface.py
    ├── watus_face.html
    └── requirements.txt
```

## 🎯 Jak to działa?

### Auto-Detection (Bez Modyfikacji)
```
watus.py (Działa Normalnie)
    ↓ ZMQ Publish (tcp://127.0.0.1:7780)
    ↓ (dialog.leader, unknown_utterance)
standalone_interface.py
    ↓ WebSocket Real-time
    ↓ Browser Interface
    ↓ 🎭 Beautiful Emoticon Display
```

### Słuchane ZMQ Topics
- `dialog.leader` - gdy lider mówi
- `unknown_utterance` - nieznany mówca

### Monitorowane Log Files
- `watus.log` - główne logi
- `dialog.jsonl` - historia dialogów  
- `camera.jsonl` - detekcje z kamery

## 🛠️ Configuration

### Environment Variables
```bash
STANDALONE_INTERFACE_HOST=127.0.0.1
STANDALONE_INTERFACE_PORT=5001
STANDALONE_DEMO_MODE=true
```

### Custom Detection Patterns
W `standalone_watus_interface.py` możesz dodać własne wzorce:

```python
self.state_patterns['my_state'] = [
    r'your_custom_pattern',
    r'another_pattern'
]
```

## 🐛 Troubleshooting

### Interface się nie łączy
```bash
# Sprawdź czy watus jest uruchomiony
python watus.py

# W innym terminalu
cd standalone_interface
python start_interface.py
```

### Port 5001 zajęty
```bash
export STANDALONE_INTERFACE_PORT=5002
python start_interface.py
```

### Demo mode nie działa
Interface automatycznie wchodzi w tryb demo gdy watus nie jest dostępny.

## 📊 API Endpoints

- `GET /` - Główny interface  
- `GET /status` - Status interface'u i watus
- `GET /watus_status` - Szczegółowy status watus
- `GET /health` - Health check

## 🔄 Integration z Watus

### Zero Impact
- ❌ **Nie modyfikuje** watus.py
- ❌ **Nie dodaje** dependencies do watus  
- ❌ **Nie zmienia** workflow watus
- ✅ **Słucha tylko** - nie publikuje do watus

### Graceful Degradation
- Gdy ZMQ niedostępny → monitoruje logi
- Gdy logi niedostępne → demo mode
- Gdy wszystko niedostępne → lokalne sterowanie

## 🎮 Demo Mode

Gdy watus nie jest uruchomiony, interface automatycznie:
- Cykluje przez emotikony co 8 sekund
- Wyświetla "Demo Mode" w statusie
- Pozwala na manualne sterowanie

## 📱 Multiple Clients

Możesz otworzyć interface w wielu kartach/urządzeniach jednocześnie:
- Wszystkie klienty widzą te same emotikony
- Real-time synchronization
- Każdy może manualnie sterować

## 🚀 Advanced Usage

### Multiple Interface Instances
```bash
# Różne porty
STANDALONE_INTERFACE_PORT=5001 python start_interface.py &
STANDALONE_INTERFACE_PORT=5002 python start_interface.py &
```

### Custom State Detection
Dodaj własne wzorce do automatycznego wykrywania stanów.

### Integration z innymi systemami
Interface może być rozszerzony o webhooki, bazę danych, etc.

## 🎉 Success Checklist

- [ ] Interface dostępny na http://127.0.0.1:5001
- [ ] ZMQ connection established (status OK)
- [ ] Auto-detection emotikony working
- [ ] Manual controls functional
- [ ] WebSocket connected
- [ ] Multiple clients synchronized
- [ ] watus.py działa normalnie (bez zmian)

---

## 🤝 Quick Commands

```bash
# Sprawdź gotowość
python setup_standalone.py --help

# Setup interface  
python setup_standalone.py

# Uruchom interface
cd standalone_interface && python start_interface.py

# Uruchom watus
python watus.py

# Status check
curl http://127.0.0.1:5001/status
```

**Gotowe!** 🎭 Masz teraz piękny HTML monitor dla watus_project! 🚀