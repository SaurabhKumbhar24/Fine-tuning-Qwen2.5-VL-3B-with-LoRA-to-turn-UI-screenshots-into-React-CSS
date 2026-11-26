from datasets import load_dataset
from PIL import Image as PILImage
import io

SYSTEM_PROMPT = (
    "You are a frontend developer who converts UI screenshots into React code.\n"
    "Given a screenshot, write a single React functional component that reproduces it.\n"
    "Style elements with className and put all custom styles in the CSS block.\n"
    "End the component with an export default statement.\n"
    "Output exactly two code blocks, JSX first and CSS second, with no text before, between or after them.\n"
    "Format:\n"
    "```jsx\n...\n```\n"
    "```css\n...\n```"
)
USER_PROMPT = "Generate the React component and CSS for this UI screenshot."

def create_prompt(data):
    prompt = {
        "messages":[
            {"role":"system", "content":[{"type":"text", "text": SYSTEM_PROMPT}]},
            {"role":"user", "content":[{"type":"image"}, {"type":"text", "text": USER_PROMPT}]},
            {"role":"assistant", "content":[{"type": "text", "text": f"""```jsx\n{data["component"].strip()}\n```\n```css\n{data["style"].strip()}\n```"""}]},
        ],
        "images": [PILImage.open(io.BytesIO(data["image"])).convert("RGB")],
    }
    return prompt

def save_data():
    ds = load_dataset(
        "Flame-Code-VLM/Flame-Waterfall-React", 
        data_files="single-image/Flame-Waterfall-React-Single-Image.parquet", 
        split="train"
    )

    ds = ds.train_test_split(test_size=0.05, seed=42)

    clean_ds = ds.map(create_prompt, 
                    remove_columns=[
                      "meta_data","image","style","component",
                      "code","task_description","layout_description"
                ])

    clean_ds.save_to_disk("data/flame_clean")