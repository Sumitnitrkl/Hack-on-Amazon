import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import LinkPlaidScreen from './screens/LinkPlaidScreen';
import DashboardScreen from './screens/DashboardScreen';
import PreferencesScreen from './screens/PreferencesScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="LinkPlaid">
        <Stack.Screen name="LinkPlaid" component={LinkPlaidScreen} />
        <Stack.Screen name="Dashboard" component={DashboardScreen} />
        <Stack.Screen name="Preferences" component={PreferencesScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}