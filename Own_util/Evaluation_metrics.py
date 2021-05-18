# -*- encoding: utf-8 -*-
'''
@File    :   Evaluation_metrics.py    
@Contact :   lvxingvir@gmail.com
@License :   (C)Copyright Xing

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
04/25/2021 13:24  Xing        1.0         initial version for evaluation metrics for models
'''

import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt


from sklearn import metrics


def parse_arg():
    parser = argparse.ArgumentParser(description='Evaluation_metrics')
    parser.add_argument('--arg', help='arg desc', default='', type=str)
    parser.add_argument('--title', help='arg desc', default='test', type=str)
    parser.add_argument('--save_path', help='arg desc', default='', type=str)
    parser.add_argument('--dpi', help='arg desc', default=300, type=int)
    args = parser.parse_args()
    return args


class Evaluation_metrics(object):
    def __init__(self, y, pred, title = '',save_path = '',dpi = 300, save_fig = True, show_fig = True):

        self.y = y
        self.pred = pred

        self.title = title
        self.save_path = save_path
        self.dpi = dpi

        self.save_fig = save_fig
        self.show_fig = show_fig

    def roc_curve(self):

        fpr, tpr, thresholds = metrics.roc_curve(self.y, self.pred)

        roc_auc = metrics.auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for {}'.format(self.title))
        plt.legend(loc="lower right")
        fig_name = 'ROC_curve_{}.png'.format(self.title)
        plt.savefig(os.path.join(self.save_path,fig_name),dpi=self.dpi)
        plt.show()

    def cohen_kappa(self):
        mali = []
        beni = []
        cohen = []
        tt = np.arange(0, 1, 0.01)
        for thres in tt:
            #     print(thres)
            pred_t = self.pred > thres
            pred_t.astype(int)
            cm = metrics.confusion_matrix(self.y, pred_t)
            mali.append(cm[1, 1] / sum(cm[1, :]))
            beni.append(cm[0, 0] / sum(cm[0, :]))
            cohen.append(metrics.cohen_kappa_score(self.y, pred_t))
        # print(thresholds)
        plt.figure(), plt.plot(tt, mali, label='TPR'), plt.plot(tt, beni, label='TNR'), plt.plot(tt, cohen,
                                                                                                 label='cohen')
        plt.xlabel('Thresholds')
        plt.ylabel('TPR/TNR')

        plt.legend(loc="lower right")

        net_max = cohen.index(max(cohen))
        plt.title('TPR/TNR with Thresholds \n (with max_cohen_kappa_score of {} at {})'.format(max(cohen), tt[net_max]))
        plt.scatter(tt[net_max], cohen[net_max], color='b')

        fig_name = 'Cohen_kappa_{}.png'.format(self.title)
        if self.save_fig:
            plt.savefig(os.path.join(self.save_path,fig_name),dpi=self.dpi)
        if self.show_fig:
            plt.show()

        return max(cohen),tt[net_max]

    def confusion_matrix(self,target_names,
                         threshold,cmap = None,normalize = True):
        '''

        Args:
            y:
            pred:
            target_names: classification names
            threshold: threshold for the binary classification
            cmap: 'Blue'
            normalize: whether for normalization.

        Returns: no return

        '''

        import itertools

        pred_t = self.pred > threshold
        pred_t.astype(int)

        cm = metrics.confusion_matrix(self.y, pred_t)

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(4, 3))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(self.title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

        fig_name = 'Confusion_Matrix_{}.png'.format(self.title)
        if self.save_fig:
            plt.savefig(os.path.join(self.save_path, fig_name), dpi=self.dpi)
        if self.show_fig:
            plt.show()

    def classification_report(self,target_names):
        df = metrics.classification_report(self.y,self.pred,target_names=target_names)
        print(df)
        return df


    def __call__(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    args = parse_arg()

    Evaluation_metrics(title = args.title,save_path=args.save_path,dpi= args.dpi)()