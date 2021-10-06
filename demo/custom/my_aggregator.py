from __future__ import print_function

from nvflare.apis.aggregator import Aggregator
from nvflare.apis.fl_constant import FLConstants, ShareableKey, ShareableValue
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.utils.fl_ctx_sanity_check import server_fl_ctx_sanity_check

import numpy as np
from typing import Tuple

class CustomAggregator(Aggregator):
    def __init__(self):
        super().__init__()
        self.vars = []
    
    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> Tuple[bool, bool]:
        '''
        Store shareable and update aggregator's internal state
        Args:
            shareable: information from client
            fl_ctx: context provided by workflow

        Returns:
            The first boolean indicates if this shareable is accepted.
            The second boolean indicates if aggregate can be called.
        '''
        shared_fl_context = fl_ctx.get_prop(FLConstants.PEER_CONTEXT)
        self.vars.append(shared_fl_context)
        return True, False
    
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        '''
        Called when workflow determines to generate shareable to send back to clients

        Args:
            fl_ctx (FLContext): context provided by workflow

        Returns:
            Shareable: the average of accepted shareables from clients
        '''
        server_fl_ctx_sanity_check(fl_ctx)
        current_round = fl_ctx.get_prop(FLConstants.CURRENT_ROUND)
        vars_to_aggregate = self.vars[0].get_prop(FLConstants.SHAREABLE)[ShareableKey.MODEL_WEIGHTS].keys()
        
        aggregated_model = {}
        for v_name in vars_to_aggregate:
            np_vars = []
            for client_ctx in self.vars:
                data = client_ctx.get_prop(FLConstants.SHAREABLE)[ShareableKey.MODEL_WEIGHTS]
                np_vars.append(data[v_name])
            
            # Define how you aggregate the variables here
            new = np.mean(np_vars, axis=0)
            aggregated_model[v_name] = new
        
        self.vars.clear()
        
        shareable = Shareable()
        shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHTS
        shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
        shareable[ShareableKey.MODEL_WEIGHTS] = aggregated_model
        return shareable
        
    