import os
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.python.lib.io import file_io

import trainer.model as model


CHECKPOINT_PATH = "checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5"
MODEL = "model.hdf5"

class ContinuousEval(Callback):
  """Continuous eval callback to evaluate the checkpoint once
     every so many epochs.
  """
  pass

def dispatch(train_files,
             eval_files,
             job_dir,
             batch_size,
             num_epochs,
             checkpoint_epochs):
    fst_model = model.model()

    try:
      os.makedirs(job_dir)
    except:
      pass

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = CHECKPOINT_PATH
    if not job_dir.startswith("gs://"):
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
      with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
          output_f.write(input_f.read())