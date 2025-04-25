from datetime import datetime
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import History
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # 设置后端，避免需要GUI
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# 创建Flask应用
app = Flask(__name__)

# 确保static文件夹存在
if not os.path.exists('static'):
    os.makedirs('static')

# 读取数据集
print("加载数据集...")
df = pd.read_csv("spam_assassin.csv")  # 1表示垃圾邮件，0表示正常邮件
# # 下载nltk数据（如果第一次使用的话）
# nltk.download('punkt')
# nltk.download('punkt_tab')
# 分词
df['tokens'] = df['text'].apply(word_tokenize)

# 使用TF-IDF进行向量化
print("特征向量化...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['text']).toarray()  # 转换为稠密矩阵
y = df['target']  # 0=非钓鱼，1=钓鱼）

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义深度学习模型
def build_model():
    model = Sequential([
        Dense(128, activation='relu', input_dim=X.shape[1]),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # 输出层，二分类问题
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 训练并保存模型，返回训练历史
def train_and_save_model():
    model = build_model()
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test),
        epochs=5, 
        batch_size=32,
        verbose=1
    )
    model.save('phishing_model.h5')
    
    # 评估模型
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算各种评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # 保存训练历史和指标
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('static/training_history.csv', index=False)
    
    with open('static/model_metrics.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    
    # 保存ROC曲线数据
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    np.savez('static/roc_data.npz', fpr=fpr, tpr=tpr, auc=roc_auc)
    
    # 保存PR曲线数据
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    np.savez('static/pr_data.npz', precision=precision, recall=recall)
    
    # 保存测试预测结果用于可视化
    np.savez('static/test_predictions.npz', y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba[:,0])
    
    return history, metrics

# 创建可视化图表
def create_visualizations():
    print("生成可视化图表...")
    
    # 1. 训练历史曲线（Loss和Accuracy）
    if os.path.exists('static/training_history.csv'):
        hist_df = pd.read_csv('static/training_history.csv')
        
        # Loss曲线
        plt.figure(figsize=(10, 6))
        plt.plot(hist_df['loss'], label='Training Loss')
        if 'val_loss' in hist_df.columns:
            plt.plot(hist_df['val_loss'], label='Validation Loss')
        plt.title('模型训练过程中的损失变化')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig('static/loss_curve.png')
        plt.close()
        
        # Accuracy曲线
        plt.figure(figsize=(10, 6))
        plt.plot(hist_df['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in hist_df.columns:
            plt.plot(hist_df['val_accuracy'], label='Validation Accuracy')
        plt.title('模型训练过程中的准确率变化')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig('static/accuracy_curve.png')
        plt.close()
    
    # 2. 混淆矩阵
    if os.path.exists('static/confusion_matrix.npy'):
        cm = np.load('static/confusion_matrix.npy')
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['正常邮件', '钓鱼邮件'],
                    yticklabels=['正常邮件', '钓鱼邮件'])
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig('static/confusion_matrix.png')
        plt.close()
    
    # 3. ROC曲线
    if os.path.exists('static/roc_data.npz'):
        roc_data = np.load('static/roc_data.npz')
        fpr, tpr = roc_data['fpr'], roc_data['tpr']
        roc_auc = float(roc_data['auc'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率 (False Positive Rate)')
        plt.ylabel('真正例率 (True Positive Rate)')
        plt.title('接收者操作特征曲线 (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('static/roc_curve.png')
        plt.close()
    
    # 4. PR曲线
    if os.path.exists('sstatic/pr_data.npz'):
        pr_data = np.load('static/pr_data.npz')
        precision, recall = pr_data['precision'], pr_data['recall']
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title('精确率-召回率曲线')
        plt.grid(True)
        plt.savefig('static/pr_curve.png')
        plt.close()
    
    # 5. 性能指标条形图
    if os.path.exists('static/model_metrics.txt'):
        metrics = {}
        with open('static/model_metrics.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                metrics[key] = float(value)
        
        plt.figure(figsize=(0, 6))
        metrics_labels = {
            'accuracy': '准确率', 
            'precision': '精确率', 
            'recall': '召回率', 
            'f1_score': 'F1得分'
        }
        
        labels = [metrics_labels.get(k, k) for k in metrics.keys()]
        plt.bar(labels, metrics.values(), color=['blue', 'green', 'red', 'purple'])
        plt.title('模型性能指标')
        plt.ylim([0, 1.0])
        plt.grid(axis='y')
        
        # 在柱状图上显示具体数值
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
            
        plt.savefig('static/metrics_bar.png')
        plt.close()

# 训练模型并保存（如果还没有训练过模型）
def check_and_train_model():
    if not os.path.exists('phishing_model.h5'):
        print("训练新模型...")
        history, metrics = train_and_save_model()
        # 创建可视化图表
        create_visualizations()
    else:
        print("模型已存在，跳过训练...")
        # 如果需要的可视化图表不存在，则创建
        if not all(os.path.exists(f'static/{f}') for f in ['loss_curve.png', 'accuracy_curve.png', 'confusion_matrix.png']):
            create_visualizations()
    
    return load_model('phishing_model.h5')

# 加载模型
model = check_and_train_model()

# 预测函数
def predict_email(email_content):
    # TF-IDF转换
    email_tfidf = tfidf.transform([email_content])
    
    # 将scipy稀疏矩阵转换为稠密数组
    email_tfidf_dense = email_tfidf.toarray()
    
    # 预测
    prediction = model.predict(email_tfidf_dense)
    
    return prediction[0][0]  # 返回预测结果

# 添加自定义过滤器
@app.template_filter('nl2br')
def nl2br_filter(s):
    if s is None:
        return ""
    return s.replace('\n', '<br>')

@app.route('/')
def index():
    # 读取日志文件内容
    try:
        with open('prediction_results.log', 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        log_content = "暂无历史记录"
    
    return render_template('index.html', log_content=log_content)

@app.route('/pdredict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_content = request.form['email_content']
        prediction = predict_email(email_content)
        result = 'Phishing' if prediction > 0.5 else 'Not Phishing'
        
        # 获取当前时间作为时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 将结果保存到本地日志文件
        log_entry = f"[{timestamp}] 预测结果: {result} (概率: {prediction:.4f})\n"
        log_entry += f"邮件内容:\n{email_content}\n"
        log_entry += "-" * 80 + "\n\n"
        
        with open('prediction_results.log', 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # 读取日志文件内容
        try:
            with open('prediction_results.log', 'r', encoding='utf-8') as f:
                log_content = f.read()
        except FileNotFoundError:
            log_content = "暂无历史记录"
        
        return render_template('result.html', result=result, email_content=email_content, 
                              prediction_value=prediction, log_content=log_content)

@app.route('/visualize')
def visualize():
    """可视化页面，展示模型训练结果和性能指标"""
    # 读取性能指标
    metrics = {}
    if os.path.exists('static/model_metrics.txt'):
        with open('static/model_metrics.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                metrics[key] = float(value)
    
    return render_template('visualize.html', metrics=metrics)

@app.route('/retrasin', methods=['POST'])
def retrain():
    """重新训练模型"""
    try:
        # 删除旧模型
        if os.path.exists('phishsing_model.h5'):
            os.remove('phishing_model.h5')
        
        # 重新训练
        history, metrics = train_and_save_model()
        
        # 创建可视化图表
        create_visualizations()
        
        return jsonify({'status': 'success', 'message': '模型重新训练成功!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'训练失败: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)