import axios from 'axios';

const API_BASE = 'https://your-api-gateway-url'; // Replace with your actual API Gateway URL

export const createLinkToken = async (userId) => {
  const res = await axios.get(`${API_BASE}/create-link-token?userId=${userId}`);
  return res.data.link_token;
};

export const exchangePublicToken = async (publicToken, userId) => {
  return await axios.post(`${API_BASE}/exchange-token`, {
    public_token: publicToken,
    user_id: userId
  });
};

export const fetchTransactions = async (userId) => {
  const res = await axios.get(`${API_BASE}/fetch-transactions?userId=${userId}`);
  return res.data.transactions;
};

export const identifySubscriptions = async (userId, transactions) => {
  const res = await axios.post(`${API_BASE}/identify-subscriptions?userId=${userId}`, transactions);
  return res.data.identified;
};

export const suggestSubscriptionAction = async (payload) => {
  const res = await axios.post(`${API_BASE}/suggest-subscription-action`, payload);
  return res.data;
};

export const getBalance = async (userId) => {
  const res = await axios.get(`${API_BASE}/get-balance?userId=${userId}`);
  return res.data;
};

export const suggestSavings = async (payload) => {
  const res = await axios.post(`${API_BASE}/suggest-savings`, payload);
  return res.data;
};