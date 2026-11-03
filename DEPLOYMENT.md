# Deployment notes

## Local
Follow the platform-specific steps in README.md. Start with `streamlit run app.py`.

## Docker
```bash
docker build -t pm-cdss .
docker run --rm -p 8501:8501 pm-cdss
```

Open http://localhost:8501.

## Streamlit Community Cloud
Push the project folder to a GitHub repository, select `app.py` as the entry point, and deploy. Keep the CSV in `data/ai4i2020.csv`.

## Operational notes
The first launch trains the two models and can take a few minutes depending on CPU. Subsequent reruns use Streamlit's resource cache. For production, replace the reconstructed models with the validated original artifacts and run a separate validation on representative plant data.
