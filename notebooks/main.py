import mlflow
import ingest_data
import train
import score


# Create nested runs
experiment_id = mlflow.create_experiment("housing_experiment")
with mlflow.start_run(run_name="PARENT_RUN") as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(run_name="ingest_data", nested=True) as child_run1:
        mlflow.log_param("ingest_data", "yes")
        ingest_data.data_preparation()

    with mlflow.start_run(run_name="train", nested=True) as child_run2:
        mlflow.log_param("train", "yes")
        train.train()

    with mlflow.start_run(run_name="score", nested=True) as child_run3:
        mlflow.log_param("score", "yes")
        score.score()

print("parent run:")
print("run_id: {}".format(parent_run.info.run_id))
print("ingest_run_id: {}".format(child_run1.info.run_id))
print("train_run_id: {}".format(child_run2.info.run_id))
print("score_run_id: {}".format(child_run3.info.run_id))
