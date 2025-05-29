import mongoose from "mongoose";

const connect = async () => {
  try {
    await mongoose.connect("mongodb://127.0.0.1:27017/alpaca");
    console.log("Database connected!");
  } catch (err) {
    console.error("Database connection error:", err);
  }
};

export default connect;
