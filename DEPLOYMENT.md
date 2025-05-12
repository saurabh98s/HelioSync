# Model Deployment

This document explains how to deploy models in the Federated Learning system, including deployment to Hugging Face Hub.

## Prerequisites

To deploy models, you need:

1. A completed federated learning project with a final model
2. Proper configuration of deployment credentials in `config.env`
3. The required Python packages

### Required Packages

For Hugging Face Hub deployment, you need:
```
pip install huggingface_hub
```

## Configuration

The system uses a `config.env` file for sensitive configuration. To set up Hugging Face deployment, you need to:

1. Create or update your `config.env` file with your Hugging Face API token:
   ```
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   ```

2. You can generate a Hugging Face token by:
   - Going to https://huggingface.co/settings/tokens
   - Clicking "New token"
   - Selecting "Write" access
   - Copying the token to your `config.env` file

## Deployment Types

The system supports multiple deployment methods:

### File Download Deployment

This prepares the model for direct file download in appropriate formats (h5, pt, etc.).

1. Click "Deploy for Download" on the model page
2. Once deployed, use the "Download" button to get the model file

### Hugging Face Hub Deployment

This deploys your model to Hugging Face Hub for sharing and inference.

1. Click "Deploy to Hugging Face" on the model page
2. The system will:
   - Package your model with appropriate metadata
   - Create a repository on Hugging Face Hub
   - Upload the model and metadata
   - Provide a link to your deployed model

After deployment, you'll see a link to your model on Hugging Face Hub, where you can:
- View model details and metrics
- Share the model with others
- Access the model via the Hugging Face API

## Deployment Status

You can see the deployment status of your models in the project models list and on each model's detail page.

## Troubleshooting

If you encounter issues with deployment:

1. Check the application logs for detailed error messages
2. Verify your Hugging Face token is valid
3. Ensure the model file exists and is accessible
4. Make sure you have the required Python packages installed

You can run the test script to validate your Hugging Face configuration:
```
python test_huggingface_deployment.py 