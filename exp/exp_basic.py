import os
import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {}

        if args.model == 'psdi_kd':
            try:
                from src.psdi_kd import model as psdi_kd
                self.model_dict['psdi_kd'] = psdi_kd
            except ImportError:
                # src.psdi_kd.model is legacy/unused by the RUL pipeline
                # (Exp_RUL builds its model directly from src.psdi_kd.rul_model
                # and never reads self.model_dict). Its extra deps (einops,
                # timm, pytorch_wavelets, ...) are optional for that reason.
                pass


        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        if args.is_training:
            self._log_model_parameters()
        
        
    def _log_model_parameters(self):
        """
        Log model parameters.
        """
        def count_learnable_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        def count_total_parameters(model):
            return sum(p.numel() for p in model.parameters())

        learable_params = count_learnable_parameters(self.model)
        total_params = count_total_parameters(self.model)
        print(f"Learnable model parameters: {learable_params:,}")
        print(f"Total model parameters: {total_params:,}")
        

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
