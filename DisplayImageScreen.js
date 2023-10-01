import React from 'react';
import { StyleSheet, View, Image, Text } from 'react-native';

const DisplayImageScreen = ({ route }) => {
  const { imageUri } = route.params;

  return (
    <View style={styles.container}>
      <Image source={{ uri: imageUri }} style={styles.image} />
      <View style={styles.textBox}>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
  },
  image: {
    width: 480,
    height: 480,
    resizeMode: 'contain',
  },
  textBox: {
    marginTop: 20,
  },
});

export default DisplayImageScreen;
