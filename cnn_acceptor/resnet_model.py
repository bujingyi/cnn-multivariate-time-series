from cnn_acceptor import resnet_model


def estimator_model(params):
    model_params = {'init': 'do not know how to manage this piece of code...'}
    if params.model == 'rnn':
        if params.cell == 'lstm':
            model_params['rnn'] = rnn_model.lstm_block
        if params.cell == 'gru':
            model_params['rnn'] == rnn_model.gru_block
        return rnn_model.rnn_generator(
            model_params['rnn'],
            params.state_size, 
            params.num_layers, 
            params.out_width, 
            params.data_format
        )
    
    if params.model == 'cnn':
        model_params = {'block': resnet_model.bottleneck_block, 'layers': [3, 4, 6, 3]}
        return resnet_model.resnet_generator(
            model_params['block'], 
            model_params['layers'], 
            params.out_width, 
            params.data_format
        )
