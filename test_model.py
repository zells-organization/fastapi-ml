from model import train, predict, convert

ticker = "LHA.DE"

train(ticker)
prediction_list = predict(ticker)
result = convert(prediction_list)
print(result)
