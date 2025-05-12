"""
Project Forms

This module provides forms for project creation and management.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, IntegerField, SubmitField, SelectMultipleField
from wtforms.validators import DataRequired, Length, NumberRange, ValidationError, Optional

class ProjectForm(FlaskForm):
    """Form for creating and editing projects."""
    name = StringField('Project Name', validators=[
        DataRequired(),
        Length(min=3, max=120, message='Project name must be between 3 and 120 characters.')
    ])
    
    description = TextAreaField('Description', validators=[
        Optional(),
        Length(max=500, message='Description cannot exceed 500 characters.')
    ])
    
    dataset_name = StringField('Dataset Name', validators=[
        DataRequired(),
        Length(min=2, max=50, message='Dataset name must be between 2 and 50 characters.')
    ])
    
    framework = SelectField('Framework', choices=[
        ('tensorflow', 'TensorFlow'),
        ('pytorch', 'PyTorch'),
        ('scikit-learn', 'Scikit-Learn')
    ], validators=[DataRequired()])
    
    min_clients = IntegerField('Minimum Clients', validators=[
        DataRequired(),
        NumberRange(min=1, message='Minimum number of clients must be at least 1.')
    ], default=2)
    
    rounds = IntegerField('Training Rounds', validators=[
        DataRequired(),
        NumberRange(min=1, max=100, message='Training rounds must be between 1 and 100.')
    ], default=5)
    
    submit = SubmitField('Create Project')

class ModelDeploymentForm(FlaskForm):
    """Form for deploying models."""
    deploy_type = SelectField('Deployment Type', choices=[
        ('api', 'REST API'),
        ('download', 'File Download'),
        ('huggingface', 'Hugging Face Hub')
    ], validators=[DataRequired()])
    
    submit = SubmitField('Deploy Model')

class ClientAssignmentForm(FlaskForm):
    """Form for assigning clients to projects."""
    clients = SelectMultipleField('Select Clients', validators=[DataRequired()], coerce=int)
    submit = SubmitField('Assign Clients')
    
    def __init__(self, *args, available_clients=None, **kwargs):
        super(ClientAssignmentForm, self).__init__(*args, **kwargs)
        if available_clients:
            self.clients.choices = [(client.id, client.name) for client in available_clients] 