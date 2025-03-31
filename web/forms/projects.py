"""
Project Forms

This module provides forms for project management.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, IntegerField, SubmitField, RadioField
from wtforms.validators import DataRequired, Length, NumberRange, Optional

class ProjectForm(FlaskForm):
    """Form for creating a new project."""
    name = StringField('Project Name', validators=[
        DataRequired(),
        Length(min=3, max=120, message='Project name must be between 3 and 120 characters.')
    ])
    description = TextAreaField('Description', validators=[
        Length(max=500, message='Description cannot exceed 500 characters.')
    ])
    dataset_name = SelectField('Dataset', validators=[DataRequired()], 
                             choices=[
                                 ('mnist', 'MNIST - Handwritten Digits'),
                                 ('cifar10', 'CIFAR-10 - Image Classification'),
                                 ('sentiment', 'Sentiment Analysis - Text Classification')
                             ])
    framework = SelectField('Framework', validators=[DataRequired()],
                          choices=[
                              ('tensorflow', 'TensorFlow'),
                              ('pytorch', 'PyTorch')
                          ])
    min_clients = IntegerField('Minimum Clients', validators=[
        DataRequired(),
        NumberRange(min=1, max=100, message='Must be between 1 and 100 clients.')
    ], default=2)
    rounds = IntegerField('Training Rounds', validators=[
        DataRequired(),
        NumberRange(min=1, max=100, message='Must be between 1 and 100 rounds.')
    ], default=5)
    submit = SubmitField('Create Project')

class ModelDeploymentForm(FlaskForm):
    """Form for deploying a model."""
    deploy_type = RadioField('Deployment Type', validators=[DataRequired()],
                           choices=[
                               ('api', 'Deploy as API'),
                               ('download', 'Download Model File')
                           ], default='api')
    submit = SubmitField('Deploy') 