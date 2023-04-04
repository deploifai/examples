#!/bin/bash

python -c "import torch; print(f'Is cuda available: {torch.cuda.is_available()}');"

uvicorn --host 0.0.0.0 --port $PORT app:app
