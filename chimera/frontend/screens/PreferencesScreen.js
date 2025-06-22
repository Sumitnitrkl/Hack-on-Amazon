import React, { useState } from 'react';
import { View, Text, TextInput, Button, Alert } from 'react-native';
import { suggestSavings, getBalance } from '../utils/api';

export default function PreferencesScreen() {
  const userId = 'test_user_123';
  const [buffer, setBuffer] = useState('1000');
  const [goal, setGoal] = useState('Trip to Japan');
  const [sweepSuggestion, setSweepSuggestion] = useState(null);

  const handleSuggest = async () => {
    try {
      const accounts = await getBalance(userId);
      const primary = accounts.find(acc => acc.subtype === 'checking');
      if (!primary) throw new Error("No checking account found.");

      const res = await suggestSavings({
        user_id: userId,
        balance: primary.balances.available,
        buffer: parseFloat(buffer),
        goal: goal
      });

      if (res.sweep) setSweepSuggestion(res.sweep);
      else Alert.alert("Info", "No excess funds to sweep right now.");
    } catch (err) {
      Alert.alert("Error", err.message);
    }
  };

  return (
    <View style={{ padding: 20 }}>
      <Text style={{ fontSize: 20, marginBottom: 10 }}>Savings Preferences</Text>

      <TextInput
        style={{ borderWidth: 1, marginBottom: 10, padding: 8 }}
        keyboardType="numeric"
        placeholder="Buffer Amount (e.g. 1000)"
        value={buffer}
        onChangeText={setBuffer}
      />

      <TextInput
        style={{ borderWidth: 1, marginBottom: 10, padding: 8 }}
        placeholder="Savings Goal (e.g. Trip to Japan)"
        value={goal}
        onChangeText={setGoal}
      />

      <Button title="Suggest Savings" onPress={handleSuggest} />

      {sweepSuggestion && (
        <View style={{ marginTop: 20 }}>
          <Text style={{ fontWeight: 'bold' }}>AI Suggestion:</Text>
          <Text>{sweepSuggestion}</Text>
        </View>
      )}
    </View>
  );
}