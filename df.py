from transformers import pipeline

generator = pipeline("text-to-image", model="runwayml/stable-diffusion-v1-5", device=0)
image = generator("a beautiful sunset over the mountains").images[0]
image.save("sunset.png")
print("图像已生成并保存为 sunset.png")