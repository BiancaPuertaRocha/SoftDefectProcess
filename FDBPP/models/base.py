

class GenericModelBase:
    def generic_pipeline(self, df, sampler=None, feature_selector=None, threshold=0.5, k=10, balance_first=False):
        """
        Função genérica para implementar pipelines. 
        Deve ser sobrescrita pelas classes filhas com a lógica do modelo.
        """
        raise NotImplementedError("Este método deve ser implementado na classe filha.")

    # Funções específicas que podem ser usadas em subclasses
    def model_raw(self, df):
        return self.generic_pipeline(df)

    def model_with_sampler(self, df, sampler):
        return self.generic_pipeline(df, sampler=sampler)

    def model_with_feature_selector(self, df, feature_selector):
        return self.generic_pipeline(df, feature_selector=feature_selector)

    def model_sampler_feature_selector(self, df, sampler, feature_selector):
        return self.generic_pipeline(df, sampler=sampler, feature_selector=feature_selector, balance_first=True)
    
    def model_feature_selector_sampler(self, df, sampler, feature_selector):
        return self.generic_pipeline(df, sampler=sampler, feature_selector=feature_selector)
