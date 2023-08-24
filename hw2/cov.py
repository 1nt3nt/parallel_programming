from PIL import Image
im = Image.open("result_cuda.ppm")
im.save("output.jpg")

