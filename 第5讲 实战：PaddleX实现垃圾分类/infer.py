import paddlex as pdx
model = pdx.load_model('output/mobilenetv3_small/best_model')
result = model.predict('188.jpg')
print("Predict Result: ", result)