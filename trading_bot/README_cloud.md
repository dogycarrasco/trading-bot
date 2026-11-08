# Deploy en GitHub Actions — Bot 24/7 gratis

GitHub Actions corre el bot cada 6h en sus servidores. No necesitas PC prendida ni conexión.

## Costo: $0

- GitHub free tier: 2,000 min/mes de Actions
- El bot usa ~30 min/run × 4 runs/día × 30 días = ~3,600 min/mes
- Si excedes, reduce frecuencia a cada 12h (cron `0 0,12 * * *`)

## Setup (15 minutos)

1. Crear repo público en GitHub
2. Subir el código: `git push -u origin main`
3. Activar GitHub Pages: Settings ? Pages ? Source: GitHub Actions
4. Editar `trading_bot/docs/dashboard.html` línea ~165: cambiar `TU_USUARIO` por tu usuario
5. Push y esperar 5 min

## Cómo funciona

- Workflow `trading-bot.yml`: corre cada 6h, ejecuta el bot, auto-commitea estado
- Workflow `deploy-dashboard.yml`: en cada push, publica el dashboard a GitHub Pages
- Dashboard web muestra: KPIs, posiciones, historial, log

## Frecuencia ajustable

Editar `.github/workflows/trading-bot.yml`:
- cron: '0 0,6,12,18 * * *'  # cada 6h (default)
- cron: '0 0,12 * * *'       # cada 12h (más conservador)
- cron: '0 * * * *'          # cada hora (más reactivo)

## Alternativas

- **Render.com**: gratis, web service, duerme tras 15 min inactivo
- **Oracle Cloud Free Tier**: VM ARM 24/7 gratis siempre, setup más complejo
- **Google Cloud Run**: serverless, requiere tarjeta
- **PythonAnywhere**: free tier limita CPU