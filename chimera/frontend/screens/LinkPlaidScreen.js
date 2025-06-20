import React, { useState, useEffect } from 'react';
import { View, Text, Button, ActivityIndicator, Alert } from 'react-native';
import { usePlaidLink } from 'react-native-plaid-link-sdk';
import { createLinkToken, exchangePublicToken } from '../utils/api';

export default function LinkPlaidScreen({ navigation }) {
  const [linkToken, setLinkToken] = useState(null);
  const userId = 'test_user_123';

  useEffect(() => {
    async function getToken() {
      try {
        const token = await createLinkToken(userId);
        setLinkToken(token);
      } catch (err) {
        Alert.alert("Error", "Unable to create link token.");
      }
    }
    getToken();
  }, []);

  const onSuccess = async (publicToken) => {
    try {
      await exchangePublicToken(publicToken, userId);
      Alert.alert("Success", "Plaid account linked successfully.");
      navigation.navigate('Dashboard');
    } catch (err) {
      Alert.alert("Error", "Could not exchange token.");
    }
  };

  const { open, ready } = usePlaidLink({
    token: linkToken,
    onSuccess: (success) => onSuccess(success.publicToken),
  });

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text style={{ marginBottom: 20 }}>Link your bank account using Plaid</Text>
      {linkToken ? (
        <Button title="Link Bank Account" onPress={open} disabled={!ready} />
      ) : (
        <ActivityIndicator size="large" />
      )}
    </View>
  );
}
