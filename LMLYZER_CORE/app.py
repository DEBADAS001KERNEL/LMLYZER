import cmd
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pyfiglet
import psutil
import torch
import os



class LmlyzerShell(cmd.Cmd):
    banner = pyfiglet.figlet_format("LMLYZER")
    print(banner)

    
    prompt = "lmlyzer> "

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.task=None
        self.models = {}


  

    def do_load_model(self, arg):
        
        parts = arg.split(maxsplit=3)
        if len(parts) != 3:
            print("Usage: load_model <task> <hf_model_name> <custom_name>")
            return

        task, hf_model_name, custom_name = parts

        try:
            print(f"Loading model '{hf_model_name}' for task '{task}' as '{custom_name}'...")
            start_time = time.time()

            pipe = pipeline(task=task, model=hf_model_name)
            self.models[custom_name] = pipe  # Save with custom name

            total_time = time.time() - start_time
            print(f"Model '{custom_name}' loaded successfully!")
            print(f"oad time: {total_time:.2f} seconds")

        except Exception as e:
            print(f"Failed to load model: {e}")

    def do_run_model(self, arg):
        
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            print("Usage: run_model <custom_name> <text>")
            return

        custom_name, text = parts

        if custom_name not in self.models:
            print(f"No model found with the name '{custom_name}'")
            return

        pipe = self.models[custom_name]
        result = pipe(text)
        print(" Output:", result)

    def do_resource(self, arg):
        try:
           process = psutil.Process(os.getpid())
           mem_bytes = process.memory_info().rss
           mem_mb = mem_bytes / 1024 / 1024
           print(f"CPU Memory used (by this app): {mem_mb:.2f} MB")

           virtual_mem = psutil.virtual_memory()
           print(f"Total system RAM: {virtual_mem.total / (1024**3):.2f} GB")
           print(f"Available RAM: {virtual_mem.available / (1024**3):.2f} GB")

           if torch.cuda.is_available():
             gpu_mem = torch.cuda.memory_allocated()
             gpu_mem_mb = gpu_mem / 1024 / 1024
             print(f" GPU Memory allocated: {gpu_mem_mb:.2f} MB")
           else:
              print("GPU not available (using CPU only).")

        except Exception as e:
          print(f"Error checking resources: {e}")

    def do_show_models(self,arg):
        print("Available Models _ ......")
        print(self.models)  


    def do_compare(self, arg):
       parts = arg.strip().split()
       if len(parts) != 2: # split the arhument in two parts 
           print("Usage: compare <model_name1> <model_name2>")
           return

       name1, name2 = parts

       if name1 not in self.models or name2 not in self.models:
        print(f" One or both models '{name1}' and '{name2}' not found.")
        return

       pipe1 = self.models[name1]
       pipe2 = self.models[name2]
       try:
        

         def measure(pipe, input_text, name):
            print(f"\n🔍 Running model '{name}'...")
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            start = time.time()

            output = pipe(input_text)

            end = time.time()
            memafter = process.memory_info().rss / 1024 / 1024
            memused = memafter - mem_before

            print(f" Memory used: {memused:.2f} MB")
            print(f" Time taken: {end - start:.2f} seconds")
            print(f" Output: {output}")

            if torch.cuda.is_available():
                gpumem = torch.cuda.memory_allocated() / 1024 / 1024
                print(f"GPU Memory allocated: {gpumem:.2f} MB")

        # Test input depending on pipeline type
         input_text = "AI is good, but can be bad"

         print("Comparing models:")
         measure(pipe1, input_text, name1)
         measure(pipe2, input_text, name2)

       except Exception as e:
        print(f"Error  {e}")
        
 

    
      



    def do_load_models(self, arg):
        
        model_names = arg.split()
        if not model_names:
            print("Please provide one or more model names  : ")
            return
        for model_name in model_names:
            self._load_model_with_timer(model_name)

    def _load_model_with_timer(self, model_name):
        try:
            print(f"Loading model: {model_name}")
            start_time = time.time()

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _ = pipeline("text-classification", model=model, tokenizer=tokenizer)

            elapsed = time.time() - start_time
            print(f"Model '{model_name}' loaded successfully!")
            print(f"Load time: {elapsed:.2f} seconds\n")
        except Exception as e:
            print(f"Failed to load model '{model_name}': {e}\n")
                    

   

def main():
    LmlyzerShell().cmdloop()
