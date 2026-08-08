# Verificador de Identidad

## Requisitos
- Windows 10/11
- Python 3.11 (se instala automaticamente)
- Google Chrome
- GPU NVIDIA (opcional, mejora rendimiento)

## Instalacion rapida

1. Copiar esta carpeta en la computadora destino
2. Ejecutar `install.ps1`:
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process
   .\install.ps1
   ```
3. Instalar extension de Chrome:
   - Abrir `chrome://extensions`
   - Activar "Modo desarrollador"
   - Click "Cargar extension descomprimida"
   - Seleccionar carpeta `extension-id-verifier`

## Uso

```powershell
.\run.ps1
```

El backend arranca en http://localhost:8000

## Archivos
- `main.py` - Backend FastAPI
- `extension-id-verifier/` - Extension de Chrome
- `install.ps1` - Instalador
- `run.ps1` - Lanzador del backend
- `requirements.txt` - Dependencias Python
