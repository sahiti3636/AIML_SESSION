from gradio_client import Client, handle_file

client = Client("linoyts/Qwen-Image-Edit-2511-Fast")

result = client.predict(
    prompt="Recolor the eyes of the cat to ice blue",
    images=[
        {
            "image": handle_file("./cat.png"),
            "caption": None
        }
    ],
    seed=0,
	randomize_seed=True,
	true_guidance_scale=1, # controls adherence to prompt lower value(< 3) -> less adherence more creativity 
	num_inference_steps=4, # higher values >25 produce high quality but increse time and resource
	height=256,
	width=256,
	rewrite_prompt=True,
	api_name="/infer"
)

print(result)
