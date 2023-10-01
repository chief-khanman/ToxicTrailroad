import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, View, Text } from 'react-native';
import { Camera } from 'expo-camera';
import Button from './src/components/Button';

const CameraScreen = ({ navigation }) => {
   const [hasCameraPermission, setHasCameraPermission] = useState(null);
   const [image, setImage] = useState(null);
   const [type, setType] = useState(Camera.Constants.Type.back);
   const [flash, setFlash] = useState(Camera.Constants.FlashMode.off);
   const cameraRef = useRef(null);

   useEffect(() => {
     (async () => {
       MediaLibrary.requestPermissionsAsync();
       const cameraStatus = await Camera.requestCameraPermissionsAsync();
       setHasCameraPermission(cameraStatus.status === 'granted');

     })();

   }, [])
  const takePicture = async () => {
    if (cameraRef) {
      try {
        const data = await cameraRef.current.takePictureAsync();
        console.log(data);
        // Save image
        navigation.navigate('DisplayImage', { imageUri: data.uri });
      } catch (e) {
        console.log(e);
      }
    }
  };

  return (
    <View style={styles.container}>
      <Camera style={styles.camera} type={type} flashMode={flash} ref={cameraRef}>
        <Text>hello</Text>
      </Camera>
      <View>
        <Button title={'Take a picture'} icon="camera" onPress={takePicture} />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
     flex: 1,
     backgroundColor: '#000',
     justifyContent: 'center',
     paddingBottom: 20 
   },
   camera: {
     flex:1,
     borderRadius: 20,
   }
});

export default CameraScreen;
