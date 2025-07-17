import os
import sys

sys.path.append(".")
sys.path.append("..")
# FIXME: if has better way to handle, MPIPoolExecutor will raise a import error for some test cases somehow
common_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common")
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{parent_dir}:{common_dir}:{pythonpath}"
