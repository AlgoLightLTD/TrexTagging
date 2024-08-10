export const environment = {
    production: false,
    firebaseConfig: {
        apiKey: "AIzaSyCOj5W9z5voOcP4OC8yHNri0McB52QpQOE",
        authDomain: "t-rex-client.firebaseapp.com",
        projectId: "t-rex-client",
        storageBucket: "t-rex-client.appspot.com",
        messagingSenderId: "241521175107",
        appId: "1:241521175107:web:f3ab71077701a73e7302f6",
        measurementId: "G-JGSC2E5BM6"
    }
};

export const SERVER = environment.production ? ["https://", "-r7ghkzrisa-uc.a.run.app"] : ["http://127.0.0.1:8000/",""];