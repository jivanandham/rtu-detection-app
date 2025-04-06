import axios from 'axios';

export const testGoogleMapsAPI = async (apiKey) => {
  try {
    // Test with a simple geocoding request
    const response = await axios.get(
      `https://maps.googleapis.com/maps/api/geocode/json?address=New+York&key=${apiKey}`
    );

    if (response.data && response.data.status === 'OK') {
      return {
        success: true,
        message: 'API key is working correctly',
        data: response.data,
      };
    } else {
      return {
        success: false,
        message: `API request failed with status: ${response.data.status}`,
        error: response.data.error_message || 'Unknown error',
      };
    }
  } catch (error) {
    return {
      success: false,
      message: 'Failed to make API request',
      error: error.message,
    };
  }
};
