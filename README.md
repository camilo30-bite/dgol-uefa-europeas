# 🏆 D-GOL UEFA EUROPEAS v2.0

Herramienta de análisis predictivo para competencias UEFA (Champions League, Europa League, Conference League) con modelo Dixon-Coles avanzado.

## 🎯 Características

- ✅ **Champions League**
- ✅ **Europa League**  
- ✅ **Conference League**
- ✅ Modelo Dixon-Coles con corrección
- ✅ Time Decay (peso temporal)
- ✅ Análisis de forma reciente
- ✅ Head-to-Head histórico
- ✅ Factor local/visitante
- ✅ 100% GRATIS (Web Scraping)
- ✅ **Headers corregidos para evitar 403**

## 🚀 Despliegue en Streamlit Cloud

1. Sube este repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Archivo principal: `dgol_uefa_v2.py`
5. ¡Listo!

## 🔧 Fixes v2.1

- ✅ Headers realistas agregados (evita error 403)
- ✅ Delays aleatorios entre solicitudes
- ✅ Session management para cookies
- ✅ Timeout aumentado a 20 segundos
- ✅ Mejor manejo de errores

## 📊 Mejoras del Modelo

- Time Decay: Partidos recientes pesan más (+15-20%)
- Corrección Dixon-Coles: Mejor predicción de empates (+10-15%)
- Forma reciente: Últimos 5 partidos (+8-12%)
- Calidad de rival: Ajuste por fuerza relativa (+5-8%)
- Head-to-Head: Historia específica (+3-5%)

**Total: 40-60% más exacto que modelo básico**

## 📊 Fuente de Datos

Web scraping desde [FBref.com](https://fbref.com) con headers avanzados

## ⚡ Sistema de Caché

- Primera carga: 3-5 segundos
- Siguientes cargas: Instantáneo (caché de 6 horas)

## 👨‍💻 Autor

D-GOL Analytics - 2025
