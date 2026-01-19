import kfp
from kfp import dsl

### componets for the pipeline. NOT the pipeline itself ###
def data_processing_op():
    return dsl.ContainerOp(
        name = "Data Processing",
        image = "cmorris2945/my-mlops-app:latest",
        command = ["python", "src/data_processing.py"]

    )



def model_training_op():
    return dsl.ContainerOp(
        name = "Model Training",
        image = "cmorris2945/my-mlops-app:latest",
        command = ["python", "src/model_training.py"]

    )

## Actual Pipeline here...

@dsl.pipeline(
    name = "MLOPS Pipeline",
    description = "This is the KubeFlow Pipeline for the cancer researc project...."
)

def mlops_pipeline():
    data_processing = data_processing_op()
    model_training = model_training_op().after(data_processing)


## RUN the pipeline...

if __name__ =="__main__":
    kfp.compiler.Compiler().compile(
        mlops_pipeline, "mlops_pipeline.yaml"
    )

