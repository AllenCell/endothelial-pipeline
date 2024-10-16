import os
import sys
import uuid
import torch
from datetime import datetime
from contextlib import contextmanager

class TrainingExperiment():

    def __init__(self, debug=False):
        self._debug = debug
        self._architecture = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_debug(self):
        return self._debug

    def get_device(self):
        return self._device

    def set_architecture(self, archtecture):
        self._architecture = archtecture
        self._architecture.to(self.get_device())
        return

    def get_architecture(self):
        return self._architecture

    def initialize_new_experiment(self):
        self._mid = str(uuid.uuid4())[:8]
        self._log_path = os.path.join(self.get_output_dir(), "code.log")
        self._var_path = os.path.join(os.path.dirname(__file__), "const.py")
        self._dts_path = os.path.join(os.path.dirname(__file__), "dataset.py")
        self._trn_path = os.path.join(os.path.dirname(__file__), "trainer.py")
        self.add_date_and_time_to_log(text="Start time")
        self.add_file_to_log(self._var_path)
        self.add_file_to_log(sys.argv[0])
        self.add_file_to_log(self._dts_path)
        self.add_file_to_log(self._trn_path)
        print(f"Experiment initialized: {self._mid}.")
        if self.get_debug():
            print("<< RUNNING IN DEBUG MODE >>")
        return

    def add_date_and_time_to_log(self, text):
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self._log_path, "a") as log_file:
            log_file.write(f'{text}: {formatted_time}\n')
        self.add_separation_to_log()
        return

    def add_separation_to_log(self, sep="="):
        os.system(f"echo '\n\n{sep*30}\n\n' >> {self._log_path}")
        return

    def add_file_to_log(self, file_path):
        os.system(f"cat {file_path} >> {self._log_path}")
        self.add_separation_to_log()
        return

    def finalize_experiment(self):
        self.add_date_and_time_to_log(text="End time")
        print(f"Experiment {self._mid} is complete.")
        return
    
    def get_output_dir(self, subfolder=None, create=True):
        output_dir = os.path.join(".output", self._mid)
        if subfolder is not None:
            output_dir = os.path.join(output_dir, subfolder)
        if create and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

@contextmanager
def run_experiment(debug):
    try:
        exp = TrainingExperiment(debug=debug)
        exp.initialize_new_experiment()
        yield exp
    finally:
        exp.finalize_experiment()

