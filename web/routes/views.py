@views_bp.route('/clients')
@login_required
def client_dashboard():
    """Client control dashboard."""
    # Get all clients for the current organization
    clients = Client.query.filter_by(organization_id=current_user.organization_id).all()
    
    # Get federated learning server instance
    fl_server = current_app.fl_server
    
    if fl_server:
        # Get current metrics
        metrics = fl_server.get_metrics()
        
        # Get client statuses
        client_statuses = fl_server.client_statuses
        
        # Get round metrics
        round_metrics = fl_server.round_metrics
    else:
        metrics = {}
        client_statuses = {}
        round_metrics = {}
    
    return render_template('clients/dashboard.html',
                         clients=clients,
                         metrics=metrics,
                         client_statuses=client_statuses,
                         round_metrics=round_metrics)

@views_bp.route('/clients/<client_id>/control', methods=['POST'])
@login_required
def control_client(client_id):
    """Control a client's training process."""
    action = request.form.get('action')
    
    if not action or action not in ['continue', 'stop', 'wait']:
        return jsonify({"error": "Invalid action"}), 400
    
    # Get the federated learning server instance
    fl_server = current_app.fl_server
    
    if not fl_server:
        return jsonify({"error": "Federated learning server not initialized"}), 500
    
    # Update client's status
    fl_server.update_client_status(client_id, action)
    
    return jsonify({
        "status": "success",
        "message": f"Client {client_id} status updated to {action}"
    })

@views_bp.route('/metrics')
@login_required
def metrics_dashboard():
    """Metrics visualization dashboard."""
    # Get federated learning server instance
    fl_server = current_app.fl_server
    
    if not fl_server:
        return render_template('metrics/dashboard.html', metrics={})
    
    # Get current metrics
    metrics = fl_server.get_metrics()
    
    return render_template('metrics/dashboard.html', metrics=metrics) 