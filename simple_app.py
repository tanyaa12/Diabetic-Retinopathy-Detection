import gradio as gr

def greet(name):
    return "Hello, " + name + "!"

# Create a simple Gradio interface
demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="Diabetic Retinopathy Detection - Test App",
    description="This is a simple test app to verify Gradio is working properly."
)

if __name__ == "__main__":
    print("Starting simple test app...")
    demo.launch()
    print("Application launched successfully!")