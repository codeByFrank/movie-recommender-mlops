-- Creates a dedicated MLflow DB + user the first time MySQL starts
CREATE DATABASE IF NOT EXISTS mlflow CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'mlflow'@'%' IDENTIFIED BY 'mlflow';
GRANT ALL PRIVILEGES ON mlflow.* TO 'mlflow'@'%';
FLUSH PRIVILEGES;