#!/bin/bash

uvicorn --host 0.0.0.0 --port $PORT app:app
