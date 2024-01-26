class Logger:

    def __init__(self):
        self.stats = {'score':[], 'epsilon':[], 'loss':[], 'iterations':[]}

    def init_stats(self):
        """Initializes a running tally of statistics"""
        stats = {
            'loss':0,
            'count':0,
            'score':0,
            'iterations':0,
            'epoch_score':0,
            'epoch_iterations':0
        }
        return stats

    def init_epoch(self, stats):
        """resets running tally of stats for an epoch"""
        stats['epoch_score'] = 0
        stats['epoch_iterations'] = 0
        return stats

    def update_running_stats(self, stats, score, loss=None):
        stats['iterations'] = stats['iterations'] + 1
        stats['epoch_score'] = stats['epoch_score'] + score
        stats['epoch_iterations'] = stats['epoch_iterations'] + 1
        if loss is not None:
            stats['loss'] = stats['loss'] + loss
            stats['count'] = stats['count'] + 1
        return stats

    def update_overall_stats(self, stats, eps, epoch, num_epochs):
        avg_score = stats['score'] / 100
        avg_its = stats['iterations'] / 100
        self.stats['score'].append(avg_score)
        if eps:
            self.stats['epsilon'].append(eps)
        try:
            loss = stats['loss'] / stats['count']
        except Exception as e:
            print(f'Error with epoch loss: {e}')
            loss = 0
        self.stats['loss'].append(loss)
        self.stats['iterations'].append(avg_its)

        print(f'---> Epoch {epoch}/{num_epochs}, Score: {avg_score}, eps: {eps}')
        print(f'-------->Loss: {loss}, Its: {avg_its}')

        early_stop = False
        score_stop = False
        iteration_stop = False
        if len(self.stats['score']) > 5:
            scores = self.stats['score']
            its = self.stats['iterations']
            if scores[-1] - scores[-5] < 100:
                score_stop = True
            if its[-1] - its[-5] < 20:
                iteration_stop = True
            if score_stop and iteration_stop:
                early_stop = False
        return early_stop

    def end_epoch(self, stats):
        stats['score'] = stats['score'] + stats['epoch_score']
        stats['iterations'] = stats['iterations'] + stats['epoch_iterations']
        return stats