from multiprocessing import freeze_support
import runpy

if __name__ == "__main__":
    freeze_support()
    runpy.run_path("train_script.py", run_name="__main__")