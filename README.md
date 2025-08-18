# Football Match Prediction API

This project provides a backend service for football match prediction, including model management, authentication, and LLM-powered analysis.

## Project Structure

```
predict-back/
├── .env
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   ├── artifacts/
│   ├── core/
│   ├── data/
│   ├── db/
│   ├── ml/
│   ├── models/
│   ├── plots/
│   ├── scripts/
│   └── services/
```

## API Overview

Base URL: `http://localhost:8000/`

### Authentication
- **POST /auth/token**: Obtain JWT access token (username & password required).
- **GET /auth/me**: Get current user info (requires token).
- **GET /auth/admin-check**: Check admin status (requires admin token).

### Prediction Endpoints
- **POST /predict/match**: Predict match outcome.
  - Modes:
    - `auto`/`server`: API computes features from DB, then predicts.
    - `features`: Caller supplies features dict (must match model meta).
- **POST /predict/enriched**: Predict with LLM analysis. Requires at least:
  - `home_team`, `away_team`, `match_date` (optional: `league`)
  - Returns: model output + LLM analysis.

### Model Management (Admin)
- **GET /admin/models/**: List models (paginated, requires admin token).
- **GET /admin/models/{model_id}**: Get model details.
- **POST /admin/models/{model_id}/activate**: Activate a model (set as active, optionally reload).
- **POST /admin/models/{model_id}/deactivate**: Deactivate a model.
- **POST /admin/models/{model_id}/reload**: Reload model artifact for given model id.

## Example: Predict Match
```json
POST /predict/match
{
  "home_team": "Man United",
  "away_team": "Fulham",
  "match_date": "2025-08-16",
  "mode": "auto"
}
```

## Environment Variables
- `GEMINI_API_KEY`: API key for Google Gemini (LLM analysis)
- `GEMINI_MODEL`: Model name (default: `gemini-2.0-flash`)
- `DATABASE_URL`: Database connection string (e.g., `postgresql://user:password@localhost/dbname`)
- `MODEL_PATH`: Filesystem path to the trained model artifact (e.g., `app/artifacts/model.pkl`)

## Setup
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Set up `.env` with required variables.
3. Run the server:
   ```sh
   uvicorn app.main:app --reload
   ```

## License
MIT
