returns', 'log_returns'], axis=1).values).to(self.config['device'])
                volatility = torch.FloatTensor(df['Volatility'].values).to(self.config['device'])
                accuracy = torch.FloatTensor(df['Accuracy'].values).to(self.config['device'])
                trend_strength = torch.FloatTensor(df['Trend_Strength'].values).to(self.config['device'])
                
                predictions, weights = model([X.unsqueeze(0)], volatility, accuracy, trend_strength)
                signals[tf] = self.process_predictions(predictions.squeeze().cpu().numpy(), df)
                dynamic_weights[tf] = weights.squeeze().cpu().numpy()
        return signals, dynamic_weights

    def process_predictions(self, predictions, df):
        signal_strength = predictions[0]
        entry_level = self.determine_entry_level(predictions[1], df['Trend_Strength'].iloc[-1])
        stop_loss = predictions[2] * df['ATR'].iloc[-1]
        
        return {
            'signal_strength': signal_strength,
            'entry_level': entry_level,
            'stop_loss': stop_loss
        }

    def determine_entry_level(self, raw_level, trend_strength):
        if raw_level > 0.7 and trend_strength > 0.05:
            return 2  # Strong entry
        elif raw_level > 0.5 and trend_strength > 0.03:
            return 1  # Medium entry
        elif raw_level > 0.3 and trend_strength > 0.01:
            return 0.5  # Weak entry
        else:
            return 0  # No entry

    def generate_comprehensive_signal(self, signals, dynamic_weights, trend_cons):
        s_final = sum(w * s['signal_strength'] for w, s in zip(dynamic_weights, signals.values()))
        s_comprehensive = s_final * trend_cons
        return s_comprehensive

    def determine_entry_strategy(self, s_comprehensive, entry_level, tc, risk_total):
        th1, th2, th3 = 0.7, 0.5, 0.3
        r1, r2, r3 = 0.1, 0.2, 0.3

        if s_comprehensive > th1 and entry_level > 1.5 and tc > 0 and risk_total < r1:
            return "Strong_Long"
        elif th2 < s_comprehensive <= th1 and 1 < entry_level <= 1.5 and tc > 0 and risk_total < r2:
            return "Medium_Long"
        elif th3 < s_comprehensive <= th2 and 0.5 < entry_level <= 1 and tc > 0 and risk_total < r3:
            return "Weak_Long"
        elif -th3 <= s_comprehensive <= th3 or entry_level <= 0.5 or risk_total >= r3:
            return "Neutral"
        elif -th2 < s_comprehensive <= -th3 and 0.5 < entry_level <= 1 and tc < 0 and risk_total < r3:
            return "Weak_Short"
        elif -th1 < s_comprehensive <= -th2 and 1 < entry_level <= 1.5 and tc < 0 and risk_total < r2:
            return "Medium_Short"
        elif s_comprehensive <= -th1 and entry_level > 1.5 and tc < 0 and risk_total < r1:
            return "Strong_Short"
        else:
            return "Neutral"
