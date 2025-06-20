import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, Button, ActivityIndicator, Alert } from 'react-native';
import { fetchTransactions, identifySubscriptions, suggestSubscriptionAction } from '../utils/api';

export default function DashboardScreen() {
  const userId = 'test_user_123';
  const [subscriptions, setSubscriptions] = useState([]);
  const [loading, setLoading] = useState(false);

  const processTransactions = async () => {
    setLoading(true);
    try {
      const txns = await fetchTransactions(userId);
      const subs = await identifySubscriptions(userId, txns);
      setSubscriptions(subs);
    } catch (err) {
      Alert.alert("Error", "Could not identify subscriptions.");
    } finally {
      setLoading(false);
    }
  };

  const handleActionClick = async (sub) => {
    try {
      const payload = {
        user_id: userId,
        subscription_id: sub.subscriptionId,
        service: sub.serviceName,
        amount: sub.amount
      };
      const aiResponse = await suggestSubscriptionAction(payload);
      Alert.alert(
        "AI Suggestion",
        `${aiResponse.tip}\n\nDraft Email:\n${aiResponse.email}\n\nURL: ${aiResponse.url}`
      );
    } catch (err) {
      Alert.alert("Error", "Failed to fetch AI suggestion.");
    }
  };

  useEffect(() => {
    processTransactions();
  }, []);

  return (
    <View style={{ padding: 20 }}>
      <Text style={{ fontSize: 20, marginBottom: 10 }}>Identified Subscriptions</Text>
      {loading ? (
        <ActivityIndicator size="large" />
      ) : (
        <FlatList
          data={subscriptions}
          keyExtractor={(item) => item.subscriptionId}
          renderItem={({ item }) => (
            <View style={{ marginBottom: 15, borderWidth: 1, padding: 10, borderRadius: 8 }}>
              <Text style={{ fontWeight: 'bold' }}>{item.serviceName}</Text>
              <Text>Amount: ${item.amount} / {item.recurrence}</Text>
              <Button title="Get AI Suggestions" onPress={() => handleActionClick(item)} />
            </View>
          )}
        />
      )}
    </View>
  );
}