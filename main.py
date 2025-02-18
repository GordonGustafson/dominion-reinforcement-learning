import sys

from pathlib import Path

from models.reinforce import train_reinforce_model
from models.value_function_approximation import train_value_function_approximation_model

# train_value_function_approximation_model()

output_path = Path(sys.argv[1])
train_reinforce_model(output_path=output_path)
