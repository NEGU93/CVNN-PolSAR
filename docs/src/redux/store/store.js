import { configureStore } from "@reduxjs/toolkit";
import imagesReducer from "../slices/imagesSlice";

export const store = configureStore({
  reducer: {
    imagesReducer: imagesReducer,
  },
});
