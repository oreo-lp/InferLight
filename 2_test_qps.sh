# warmup
ab -n 1 -c 1 -T "application/json" -p ./data/input.json http://localhost:8000/batch_predict

ab -n 10 -c 1 -T "application/json" -p ./data/input.json http://localhost:8000/batch_predict

ab -n 5000 -c 100 -T "application/json" -p ./data/input.json http://localhost:8000/batch_predict