<h3>Stock price predictor</h3>

Very thankful to [testdriven.io](https://testdriven.io/) and their
cool [tutorial](https://testdriven.io/blog/fastapi-machine-learning/)

Start server

`uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008`

Test model

`python test_model.py`

What's next?

    1. Containerize your environment with Docker
    2. Set up a database to save prediction results
    3. Add logging and monitoring
    4. Convert your view functions and the model prediction function into asynchronous functions
    5. Run the prediction as a background task to prevent blocking
    6. Add tests
    7. Store trained models to AWS S3, outside of Heroku's ephemeral filesystem
