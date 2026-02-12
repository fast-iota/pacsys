// ScalingHarness.java — standalone Java scaling validation harness
//
// Protocol: reads CSV from stdin, writes "OK\t<result>" or "ERR\t<message>" to stdout
//   PS,<p_index>,<input_len>,<raw>                      -> primary scale
//   PU,<p_index>,<input_len>,<value>                    -> primary unscale
//   CS,<c_index>,<const1>:...:<constN>,<x>              -> common scale
//   CU,<c_index>,<const1>:...:<constN>,<p_index>,<x>    -> common unscale

import java.io.*;

public class ScalingHarness {

    static final float FLT_MAX = 1.7014117e+38f;
    static final int MAX_BISECT = 30;
    static final int MAX_SEARCH = 1000;
    static final double TOLDIV = 131072.0;

    static final double[] lowlim = {
        -10.24, -10.0, -5.0, -2.5, 0.0, -4.295E9,
        0.0, 100.0, -FLT_MAX, 0.0, 0.0, -4.295E9, -FLT_MAX, -0.310269935,
        -4.295E9, -128.0, -128.0, 0.0, 0.0, -0.310269935, -8388608.0, 0.0,
        0.0, 0.0, -6.125E36, 0.0, -FLT_MAX, 4.0, -10.0, 0.0,
        -FLT_MAX / 500.0, 0.0, -1.0, 0.0, 0.0, -10.24, -10.24,
        0.0, 0.0, 0.0, 0.0
    };

    static final double[] uprlim = {
        10.235, 9.995, 4.998, 2.499, 65536.0,
        4.295E9, 102.35, 10.0E9, FLT_MAX, 25.0, 65535.0, 4.295E9, FLT_MAX,
        3.10269935, 4.295E9, 127.0, 127.0, 255.0, 255.0, 3.10269935,
        8388607.996, 10.0, 9999999.0, 4294967295.0, 6.125E36, 10.0,
        FLT_MAX, 20.0, 10.0, 256.0, FLT_MAX / 500.0, 10.24, 1.0, 10.235,
        0.0, 10.235, 10.235, 21.0, 4294967295.0, 5.0, 10.0
    };

    // ---- Main ----

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                writer.println("# skip");
                writer.flush();
                continue;
            }
            try {
                String[] parts = line.split(",");
                String op = parts[0].trim();
                switch (op) {
                    case "PS": {
                        int pIndex = Integer.parseInt(parts[1].trim());
                        int inputLen = Integer.parseInt(parts[2].trim());
                        int raw = Integer.parseInt(parts[3].trim());
                        double result = primaryScale(pIndex, inputLen, raw);
                        writer.println("OK\t" + result);
                        break;
                    }
                    case "PU": {
                        int pIndex = Integer.parseInt(parts[1].trim());
                        int inputLen = Integer.parseInt(parts[2].trim());
                        double value = Double.parseDouble(parts[3].trim());
                        int result = primaryUnscale(pIndex, inputLen, value);
                        writer.println("OK\t" + result);
                        break;
                    }
                    case "CS": {
                        int cIndex = Integer.parseInt(parts[1].trim());
                        double[] constants = parseConstants(parts[2].trim());
                        double x = Double.parseDouble(parts[3].trim());
                        double result = commonScale(cIndex, constants, x);
                        writer.println("OK\t" + result);
                        break;
                    }
                    case "CU": {
                        int cIndex = Integer.parseInt(parts[1].trim());
                        double[] constants = parseConstants(parts[2].trim());
                        int pIndex = Integer.parseInt(parts[3].trim());
                        double x = Double.parseDouble(parts[4].trim());
                        double result = commonUnscale(cIndex, constants, pIndex, x);
                        writer.println("OK\t" + result);
                        break;
                    }
                    default:
                        writer.println("ERR\tUnknown op: " + op);
                }
            } catch (Exception e) {
                writer.println("ERR\t" + e.getMessage());
            }
            writer.flush();
        }
        writer.flush();
    }

    static double[] parseConstants(String s) {
        if (s.isEmpty() || s.equals("_")) return new double[0];
        String[] parts = s.split(":");
        double[] result = new double[parts.length];
        for (int i = 0; i < parts.length; i++)
            result[i] = Double.parseDouble(parts[i]);
        return result;
    }

    // ---- Helpers ----

    static int checkSignedOverflow(int value, int size) throws Exception {
        if (size == 1 && (value < -128 || value > 127))
            throw new Exception("Overflow");
        if (size == 2 && (value < -32768 || value > 32767))
            throw new Exception("Overflow");
        return value;
    }

    static int checkOverflow(int value, int size) throws Exception {
        if (size == 1 && (value < 0 || value > 255))
            throw new Exception("Overflow");
        if (size == 2 && (value < 0 || value > 65535))
            throw new Exception("Overflow");
        return value;
    }

    static int checkUnsignedOverflow(int value, int size) throws Exception {
        if (size == 1 && (value < 0 || value > 255))
            throw new Exception("Overflow");
        if (size == 2 && (value < 0 || value > 65535))
            throw new Exception("Overflow");
        return value;
    }

    static short ICSHFT(short value, int nbits) {
        long x = 0, y = 0;
        int n = Math.abs(nbits);
        if (nbits > 0) {
            x = value << n;
            y |= ((0xff00 & x) >> 16) & 0xffff;
            x |= y & 0xffff;
        } else {
            x = value >> n;
            y = (~(0xffff >> n)) & 0xffff & value;
            x |= y & 0xffff;
        }
        return (short) x;
    }

    static int mod(int x, int y) {
        return x - (x / y * y);
    }

    // ---- Primary Scale (raw -> primary) ----

    static double primaryScale(int pIndex, int inputLen, int data) throws Exception {
        int x;
        switch (inputLen) {
            case 1: x = (byte) (data & 0xff); break;
            case 2: x = (short) (data & 0xffff); break;
            case 4: x = (int) (data & 0xffffffff); break;
            default: throw new Exception("Invalid scaling length");
        }

        int x1, x2, y, int_temp;
        long unsigned_x;
        double flt_data;

        switch (pIndex) {
        case 0:
            return x / 3200.0;
        case 2:
            return x / 3276.8;
        case 4:
            return x / 6553.6;
        case 6:
            return x / 13107.2;
        case 8:
            return x + 32768.0;
        case 10:
            return (double) x;
        case 12:
            return x / 320.0;
        case 14:
            flt_data = x & 0x03FF;
            int_temp = (int) (x & 0x1C00);
            int_temp >>= 10;
            for (int i = 0; i < int_temp; i++)
                flt_data *= 10.0;
            return flt_data;
        case 16:
            return Float.intBitsToFloat(x);
        case 18:
            return x * 0.0010406;
        case 20:
            flt_data = x;
            if (flt_data >= 0.0) return flt_data;
            if (inputLen != 1) flt_data += 65536.0;
            else flt_data += 256.0;
            return flt_data;
        case 22:
            x1 = (((x & 0xffff0000) >> 16) & 0x0000ffff);
            x2 = (((x & 0x0000ffff) << 16) & 0xffff0000);
            y = x1 | x2;
            return Float.intBitsToFloat(y) / (float) 4.0;
        case 24:
            x1 = (((x & 0xffff0000) >> 16) & 0x0000ffff);
            x2 = (((x & 0x0000ffff) << 16) & 0xffff0000);
            y = x1 | x2;
            return Float.intBitsToFloat(y);
        case 26:
            x >>= 8;
            y = x & 0xFF;
            return (y / 82.1865) - 0.310269935;
        case 28:
            return ((x >> 16) & 0x0000ffff) | (x << 16);
        case 30:
            return (double) (byte) x;
        case 32:
            x >>= 8;
            return (double) (byte) x;
        case 34:
            return (double) (x & 0xFF);
        case 36:
            x >>= 8;
            return (double) (x & 0xFF);
        case 38:
            y = x & 0xFF;
            return (y / 82.1865) - 0.310269935;
        case 40:
            return x / 256.0;
        case 42:
            x &= 0xFFFF;
            return x / 6553.6;
        case 44:
            flt_data = 0.0;
            for (int i = 0; i < 7; i++) {
                x <<= 4;
                int_temp = (int) ((x >> 28) & 0xF);
                flt_data *= 10.0;
                flt_data += int_temp;
            }
            return flt_data;
        case 46:
            if (x >= 0) unsigned_x = x;
            else unsigned_x = x + 4294967296L;
            return (double) unsigned_x;
        case 48:
            return Float.intBitsToFloat(x) / .036;
        case 50:
            flt_data = Float.intBitsToFloat(x);
            if (flt_data < -10.24) flt_data = -10.24;
            if (flt_data > 10.235) flt_data = 10.235;
            return flt_data;
        case 52:
            if (inputLen == 2) {
                y = (((x & 0xff00) >> 8) & 0x00ff) | ((x << 8) & 0xff00);
                return (double) (short) y;
            } else if (inputLen == 4) {
                y = (x << 24) & 0xff000000;
                y |= ((x & 0x0000ff00) << 8) & 0x00ff0000;
                y |= ((x & 0x00ff0000) >> 8) & 0x0000ff00;
                y |= ((x & 0xff000000) >> 24) & 0x000000ff;
                return (double) (int) y;
            }
            return 0.0;
        case 54:
            if ((inputLen != 2) || (((short) data & 0x8000) != 0))
                throw new Exception("Corrupt data");
            return x * 0.0004882961516 + 4.0;
        case 56:
            if (inputLen != 2)
                throw new Exception("Must be two byte quantity");
            unsigned_x = data & 0x0000ffff;
            return ((double) unsigned_x - 32768.0) / 3276.8;
        case 58:
            unsigned_x = data & 0xffffffffL;
            return unsigned_x / 256.0;
        case 60:
            return 500.0 * Float.intBitsToFloat(x);
        case 62:
            return x / 6400.0;
        case 64:
            if (inputLen == 1) return x / 128.0;
            else if (inputLen == 2) return x / 32768.0;
            else return x / 2147483648.0;
        case 66:
            if (inputLen != 2)
                throw new Exception("Must be two byte quantity");
            if (x <= 0.0)
                throw new Exception("Corrupt data");
            return x / 3200.0;
        case 68:
            throw new Exception("Alternate Scaling Required");
        case 70:
            return x / 1000.0;
        case 72:
            if (inputLen != 2)
                throw new Exception("Must be two byte quantity");
            unsigned_x = data & 0x0000ffff;
            return ((double) unsigned_x - 32768.0) / 3200.0;
        case 74:
            if (inputLen != 2)
                throw new Exception("Data length error");
            return x * 0.00064088;
        case 76:
            if (inputLen != 4)
                throw new Exception("Data length error");
            x1 = ((x & 0xFFFF0000) >> 16) & 0x0000FFFF;
            x2 = ((x & 0x0000FFFF) << 16) & 0xFFFF0000;
            y = x1 | x2;
            long unsigned_y;
            if (y >= 0) unsigned_y = y;
            else unsigned_y = (long) y + 0x100000000L;
            return (double) unsigned_y;
        case 78:
            flt_data = Float.intBitsToFloat(x);
            if (flt_data < 0.0) flt_data = 0.0;
            if (flt_data > 5.0) flt_data = 5.0;
            return flt_data;
        case 80:
            flt_data = Float.intBitsToFloat(x);
            if (flt_data < 0.0) flt_data = 0.0;
            if (flt_data > 10.0) flt_data = 10.0;
            return flt_data;
        case 82:
            if (inputLen != 2)
                throw new Exception("Must be two byte quantity");
            if ((x & 0xf000) != 0)
                throw new Exception("Corrupted input data");
            return x / 409.5;
        case 84:
            if (inputLen != 4)
                throw new Exception("Must be four byte quantity");
            x1 = (((x & 0xff000000) >> 24) & 0x000000ff);
            y = x1;
            x1 = (((x & 0x00ff0000) >> 8) & 0x0000ff00);
            y = y | x1;
            x1 = (((x & 0x0000ff00) << 8) & 0x00ff0000);
            y = y | x1;
            x1 = (((x & 0x000000ff) << 24) & 0xff000000);
            y = y | x1;
            return Float.intBitsToFloat(y);
        default:
            throw new Exception("Primary transform " + pIndex + " not found");
        }
    }

    // ---- Primary Unscale (primary -> raw) ----

    static int primaryUnscale(int pIndex, int inputLen, double value) throws Exception {
        double xx = value;
        int x1, x2, state;
        long y;
        short expon, hbit;
        double mantis;
        int temp;
        int unsigned_state;

        switch (pIndex) {
        case 0:
            return checkSignedOverflow((int) (value * 3200.0), inputLen);
        case 2:
            return checkSignedOverflow((int) (value * 3276.8), inputLen);
        case 4:
            return checkSignedOverflow((int) (value * 6553.6), inputLen);
        case 6:
            return checkSignedOverflow((int) (value * 13107.2), inputLen);
        case 8:
            return checkSignedOverflow((int) (value - 32768.0), inputLen);
        case 10:
            return checkSignedOverflow((int) value, inputLen);
        case 12:
            return checkSignedOverflow((int) (value * 320.0), inputLen);
        case 14:
            mantis = xx;
            expon = 0;
            while (mantis > 1023.0) {
                mantis /= 10.0;
                expon++;
            }
            state = (int) mantis;
            expon = ICSHFT(expon, 10);
            state = state | expon;
            state = state | 0xc000;
            return state;
        case 50:
            if ((value < -10.24) || (value > 10.235))
                throw new Exception("Out of valid range");
            // fall through
        case 16:
            return Float.floatToIntBits((float) value);
        case 18:
            return checkSignedOverflow((int) (value / 0.001040625), inputLen);
        case 20:
            if (value < 0.0)
                throw new Exception("Corrupt data");
            if (inputLen == 1) {
                if (value > 255.0) throw new Exception("Corrupt data");
            } else if (inputLen == 2) {
                if (value > 65535.0) throw new Exception("Corrupt data");
            } else if (inputLen == 4) {
                throw new Exception("Bad scale");
            }
            return (int) value;
        case 22:
            state = Float.floatToIntBits((float) (xx * 4));
            x1 = (((state & 0xffff0000) >> 16) & 0x0000ffff);
            x2 = (((state & 0x0000ffff) << 16) & 0xffff0000);
            state = x1 | x2;
            return state;
        case 24:
            state = Float.floatToIntBits((float) xx);
            x1 = (((state & 0xffff0000) >> 16) & 0x0000ffff);
            x2 = (((state & 0x0000ffff) << 16) & 0xffff0000);
            state = x1 | x2;
            return state;
        case 26:
            mantis = (float) ((xx + 0.310269935) * 82.1865 * 256.0);
            state = (int) Math.abs(mantis);
            return state;
        case 28:
            state = (int) xx;
            x1 = (((state & 0xffff0000) >> 16) & 0x0000ffff);
            x2 = (((state & 0x0000ffff) << 16) & 0xffff0000);
            state = x1 | x2;
            return state;
        case 30:
            return checkSignedOverflow((int) value, 1);
        case 32:
            state = checkSignedOverflow((short) value, 1);
            state = ICSHFT((short) state, 8);
            state = (state & 0xff00);
            return state;
        case 34:
            return checkOverflow((int) Math.abs(value), 1);
        case 36:
            state = checkSignedOverflow((short) Math.abs(value), 1);
            state = ICSHFT((short) state, 8);
            state = (state & 0xff00);
            return state;
        case 38:
            state = (int) (Math.abs((xx + 0.310269935) * 82.1865));
            state &= 0xFF;
            return state;
        case 40:
            return checkSignedOverflow((int) (value * 256.0), 1);
        case 42:
            xx *= 6553.6;
            if (xx > 32768.0) xx -= 65536.0;
            state = (int) xx;
            state = state & 0x0000ffff;
            return state;
        case 44:
            temp = (int) xx;
            state = 0;
            for (int ii = 7; ii >= 1; ii--) {
                hbit = (short) (temp / (int) Math.pow(10, ii));
                if (hbit != 0) {
                    temp = mod(temp, (int) Math.pow(10, ii));
                    state = (int) (state + hbit * Math.pow(16, ii));
                }
            }
            state = state + temp;
            return state;
        case 46:
            if (xx < 0.0)
                throw new Exception("Corrupt data");
            unsigned_state = (int) xx;
            if (unsigned_state < 0) unsigned_state = -1 * unsigned_state;
            return unsigned_state;
        case 48:
            state = Float.floatToIntBits((float) (xx * 0.036));
            return state;
        case 52:
            state = checkSignedOverflow((int) value, inputLen);
            y = 0;
            if (inputLen == 2) {
                y = (((state & 0xff00) >> 8) & 0x00ff) | ((state << 8) & 0xff00);
            } else if (inputLen == 4) {
                y = (state << 24) & 0xff000000;
                y |= ((state & 0x0000ff00) << 8) & 0x00ff0000;
                y |= ((state & 0x00ff0000) >> 8) & 0x0000ff00;
                y |= ((state & 0xff000000) >> 24) & 0x000000ff;
            }
            state = (int) (y & 0xffffffff);
            return state;
        case 54:
            if ((inputLen != 2) || (xx < 4.0) || (xx > 20.0))
                throw new Exception("Value out of range");
            state = checkOverflow((int) ((value - 4.0) / 0.0004882961516), inputLen);
            if ((state & 0x8000) != 0)
                throw new Exception("Value out of range");
            return state;
        case 56:
            if (inputLen != 2)
                throw new Exception("Data size invalid");
            return checkOverflow((int) ((xx * 3276.8) + 32768.0), inputLen);
        case 58:
            unsigned_state = (int) (xx * 256.0);
            if (unsigned_state < 0) unsigned_state = -1 * unsigned_state;
            try {
                checkUnsignedOverflow(0, inputLen);  // checks 'state' which is 0 - Java bug
                return unsigned_state;
            } catch (Exception ignore) {
                return 0;  // returns 'state' which is 0 - Java bug
            }
        case 60:
            return Float.floatToIntBits((float) (xx / 500.0));
        case 62:
            xx *= 6400.0;
            if (xx > 32768.0) xx -= 65536.0;
            state = (int) xx;
            state = state & 0x0000ffff;
            return state;
        case 64:
            if ((xx < -1.0) || (xx > 1.0))
                throw new Exception("Value out of range");
            state = 0;
            if (inputLen == 1) state = (int) (xx * 128.0);
            if (inputLen == 2) state = (int) (xx * 32768.0);
            if (inputLen == 4) state = (int) (xx * 2147483648.0);
            return state;
        case 66:
            if (xx <= 0.0)
                throw new Exception("Value out of range");
            return (int) (xx * 3200.0);
        case 68:
            return (int) xx;
        case 70:
            return checkSignedOverflow((int) (value * 1000.0), inputLen);
        case 72:
            if (inputLen != 2)
                throw new Exception("Data length invalid");
            return checkOverflow((int) ((xx * 3200.0) + 32768.0), inputLen);
        case 74:
            if ((inputLen != 2) || (xx < -21.0) || (xx > 21.0))
                throw new Exception("Data range invalid");
            return checkUnsignedOverflow((int) (xx / 0.00064088), inputLen);
        case 76:
            if (inputLen != 4)
                throw new Exception("Data length invalid");
            if ((xx >= (float) 0x100000000L) || (xx < 0))
                throw new Exception("Data range invalid");
            if (xx >= (float) 0x80000000L)
                state = (int) (xx - (float) 0x80000000L) | 0x80000000;
            else
                state = (int) xx;
            x1 = (((state & 0xffff0000) >> 16) & 0x0000ffff);
            x2 = (((state & 0x0000ffff) << 16) & 0xffff0000);
            state = x1 | x2;
            return state;
        case 78:
            if ((xx < 0) || (xx > 5.0))
                throw new Exception("Out of valid range");
            return Float.floatToIntBits((float) xx);
        case 80:
            if ((xx < 0) || (xx > 10.0))
                throw new Exception("Out of valid range");
            return Float.floatToIntBits((float) xx);
        case 82:
            if (inputLen != 2)
                throw new Exception("Data length invalid");
            if ((xx < 0) || (xx > 10.0))
                throw new Exception("Out of valid range");
            return (int) (xx * 409.5);
        case 84:
            if (inputLen != 4)
                throw new Exception("Data length invalid");
            state = Float.floatToIntBits((float) xx);
            x2 = (((state & 0xff000000) >> 24) & 0x000000ff);
            x2 = x2 | (((state & 0x00ff0000) >> 8) & 0x0000ff00);
            x2 = x2 | (((state & 0x0000ff00) << 8) & 0x00ff0000);
            x2 = x2 | (((state & 0x000000ff) << 24) & 0xff000000);
            return x2;
        default:
            throw new Exception("Primary transform " + pIndex + " not found");
        }
    }

    // ---- Common Scale (primary -> common) ----

    static double commonScale(int cIndex, double[] c, double data) throws Exception {
        double x = data;
        double flt_data, flt_temp;
        double log_e_10 = 2.302585092994045684;
        int nc = c.length;

        switch (cIndex) {
        case 0:
            return data;
        case 2:
            if (nc < 3) throw new Exception("Insufficient constants");
            if (c[1] != 0.0) return (c[0] * x / c[1]) + c[2];
            throw new Exception("Invalid constant C2 (zero)");
        case 4:
            if (nc < 2) throw new Exception("Insufficient constants");
            if (c[1] != 0.0) return (x - c[0]) / c[1];
            throw new Exception("Invalid constant C2 (zero)");
        case 6:
            if (nc < 2) throw new Exception("Insufficient constants");
            if (c[1] != 0.0) return c[0] * x / c[1];
            throw new Exception("Invalid constant C2 (zero)");
        case 8:
            if (nc < 4) throw new Exception("Insufficient constants");
            flt_temp = c[2] + c[1] * x;
            if (flt_temp != 0.0) return c[3] + (c[0] * x) / flt_temp;
            throw new Exception("Division by zero");
        case 10:
            if (nc < 3) throw new Exception("Insufficient constants");
            flt_temp = c[0] * x;
            if (flt_temp != 0.0) return c[2] + (c[1] / flt_temp);
            throw new Exception("Division by zero");
        case 12:
            if (nc < 5) throw new Exception("Insufficient constants");
            flt_data = 0;
            for (int il = 0; il < 5; il++) flt_data = c[il] + flt_data * x;
            return flt_data;
        case 14:
            if (nc < 6) throw new Exception("Insufficient constants");
            flt_temp = 0;
            for (int il = 0; il < 5; il++) flt_temp = c[il] + flt_temp * x;
            return Math.exp(flt_temp) - c[5];
        case 16:
            if (nc < 4) throw new Exception("Insufficient constants");
            if ((c[0] != 0.0) && (c[2] != 0.0))
                return c[1] * Math.exp(-x / c[0]) + c[3] * Math.exp(-x / c[2]);
            throw new Exception("Zero C1 or C3 constant");
        case 18:
            if (nc < 6) throw new Exception("Insufficient constants");
            return c[2] * Math.exp(c[1] * (x + c[0])) + c[5] * Math.exp(c[4] * (x + c[3]));
        case 20:
            if (nc < 3) throw new Exception("Insufficient constants");
            flt_temp = Math.pow(c[0] * (Math.log(x) / Math.log(10)) + c[1], 2.0);
            if (flt_temp != 0.0)
                return (Math.log(x) / Math.log(10)) / flt_temp + c[2];
            throw new Exception("Division by zero");
        case 22:
            if (nc < 2) throw new Exception("Insufficient constants");
            if (c[0] != 0.0) return c[1] * Math.pow(10.0, (x / c[0]));
            throw new Exception("Zero C1 constant");
        case 24:
            if (nc < 6) throw new Exception("Insufficient constants");
            if (x < c[0]) return c[1] * (c[2] * x + c[3]);
            return c[1] * Math.exp(c[4] * x + c[5]);
        case 26:
            if (nc < 6) throw new Exception("Insufficient constants");
            flt_data = 0;
            for (int il = 0; il < 6; il++) flt_data = c[il] + flt_data * x;
            return flt_data;
        case 28:
            if (nc < 4) throw new Exception("Insufficient constants");
            flt_temp = c[1] + c[0] * x;
            if (flt_temp != 0.0) return c[2] / flt_temp + c[3];
            throw new Exception("Division by zero");
        case 30:
            if (nc < 6) throw new Exception("Insufficient constants");
            if (x < c[0]) return c[5];
            return c[4] + (c[3] * x) + (c[2] * Math.pow(x, 2.0)) + (c[1] * Math.pow(x, 3.0));
        case 32:
            if (nc < 4) throw new Exception("Insufficient constants");
            return c[1] * Math.log(c[0] * x + c[3]) + c[2];
        case 34:
            if (nc < 4) throw new Exception("Insufficient constants");
            flt_temp = c[3] + c[2] * x;
            if (flt_temp != 0.0) return (c[1] + c[0] * x) / flt_temp;
            throw new Exception("Division by zero");
        case 36:
            if (nc < 3) throw new Exception("Insufficient constants");
            flt_temp = x + c[0];
            if (flt_temp >= 0.0) return c[1] * Math.sqrt(flt_temp) + c[2];
            throw new Exception("Square root of negative");
        case 38:
            if (nc < 6) throw new Exception("Insufficient constants");
            if (x > c[5]) {
                if (x != 0.0) {
                    if (c[2] != 0)
                        flt_temp = c[0] + (c[1] * x) + (c[2] * Math.exp(x)) + (c[3] / x) + (c[4] / Math.pow(x, 2.0));
                    else
                        flt_temp = c[0] + (c[1] * x) + (c[3] / x) + (c[4] / Math.pow(x, 2.0));
                    return Math.pow(10.0, flt_temp);
                }
                throw new Exception("Division by zero");
            }
            return 760000.0;
        case 40:
            if (nc < 6) throw new Exception("Insufficient constants");
            if (c[1] != 0.0) return (c[0] * x / c[1]) + c[2];
            throw new Exception("Zero C2 constant");
        case 42:
            if (nc < 6) throw new Exception("Insufficient constants");
            if (x < c[0]) return c[1] * Math.pow(x, 2.0) + c[2] * x + c[3];
            return c[1] * Math.exp(c[4] * x + c[5]);
        case 44:
            if (nc < 5) throw new Exception("Insufficient constants");
            if (x < c[0]) return c[1] * Math.exp(c[2] * x);
            return c[3] * Math.exp(c[4] * x);
        case 46:
            if (nc < 6) throw new Exception("Insufficient constants");
            if (x < c[0]) return c[1] * Math.exp(c[2] * Math.pow(x, 2.0) + c[3] * x);
            return c[4] * Math.exp(c[5] * x);
        case 48:
            if (nc < 3) throw new Exception("Insufficient constants");
            return c[0] * Math.pow(c[1], (1.0 / x)) * Math.pow(x, c[2]);
        case 50:
            if (nc < 2) throw new Exception("Insufficient constants");
            return c[0] * Math.acos(x / c[1]);
        case 52:
            if (nc < 5) throw new Exception("Insufficient constants");
            if (x < c[0]) return Math.exp(c[1] * x + c[2]);
            return Math.exp(c[3] * x + c[4]);
        case 54:
            if (nc < 6) throw new Exception("Insufficient constants");
            if (x < c[0]) return Math.exp(c[1] * Math.pow(x, 2.0) + c[2] * x + c[3]);
            return Math.exp(c[4] * x + c[5]);
        case 56:
            throw new Exception("Table lookup not supported");
        case 58:
            throw new Exception("Table lookup not supported");
        case 60:
            throw new Exception("Alternate Scaling Required");
        case 62:
            if (nc < 3) throw new Exception("Insufficient constants");
            if (c[0] == 0.0) throw new Exception("Zero C1 constant");
            return c[1] * (c[2] + Math.pow(10.0, (x / c[0])));
        case 64:
            if (nc < 1) throw new Exception("Insufficient constants");
            switch ((int) c[0]) {
            case 0:
                if (x <= 4.825) {
                    return 77.4 + 13.463 * x - 8.2465 * Math.pow(x, 2.0)
                        + 3.6896 * Math.pow(x, 3.0) - 0.7824 * Math.pow(x, 4.0)
                        + 0.0608 * Math.pow(x, 5.0);
                }
                return -0.5451 - 428.2927 * x - 210274.8806 * Math.pow(x, 2.0)
                    + 129913.0267 * Math.pow(x, 3.0) - 26741.8467 * Math.pow(x, 4.0)
                    + 1834.8285 * Math.pow(x, 5.0);
            case 1:
                if (x <= 0.9148) {
                    return 4.2 + 1.443 * x - 0.40755 * Math.pow(x, 2.0)
                        + 0.3005 * Math.pow(x, 3.0) - 1.5166 * Math.pow(x, 4.0)
                        + 1.3716 * Math.pow(x, 5.0);
                }
                return 4.271 + 2.89 * x - 4.11 * Math.pow(x, 2.0)
                    + 3.27 * Math.pow(x, 3.0) - 1.26 * Math.pow(x, 4.0)
                    + 0.202 * Math.pow(x, 5.0);
            default:
                throw new Exception("Invalid constant C1 for VPT");
            }
        case 66:
            if (nc < 4) throw new Exception("Insufficient constants");
            return c[0] * Math.pow(2.0, c[1] * (x + c[2])) + c[3];
        case 68:
            if (nc < 6) throw new Exception("Insufficient constants");
            if ((c[0] != 0.0) && ((c[0] * x + c[3]) <= 0.0))
                return 0.0;
            return c[5] * Math.pow(c[1] * Math.log(c[0] * x + c[3]) + c[2] * x, c[4]);
        case 70:
            if (nc < 6) throw new Exception("Insufficient constants");
            return c[0] * Math.exp(-x / c[1]) + c[2] * Math.exp(-x / c[3])
                + c[4] * Math.exp(-x / c[5]) + 4;
        case 72:
            if (nc < 6) throw new Exception("Insufficient constants");
            return c[0] * Math.pow(10.0,
                c[1] + c[2] * Math.log(x) / log_e_10
                + c[3] * Math.pow(Math.log(x) / log_e_10, 2.0)
                + c[4] * Math.pow(Math.log(x) / log_e_10, 3.0)) + c[5];
        case 74:
            if (nc < 6) throw new Exception("Insufficient constants");
            return (c[0] + c[1] * x + c[2] * x * x) / (c[3] + c[4] * x + c[5] * x * x);
        case 76:
            if (nc < 6) throw new Exception("Insufficient constants");
            if (x < c[0]) return c[1] * Math.pow(x, c[2]);
            return c[3] * Math.exp(c[4] * x + c[5]);
        case 78:
            if (nc < 4) throw new Exception("Insufficient constants");
            return c[0] * Math.pow(10, c[1] * x + c[2]) + c[3];
        case 80:
            return x;
        case 82:
            if (nc < 4) throw new Exception("Insufficient constants");
            return c[1] * Math.log(c[0] * x + c[3]) / log_e_10 + c[2];
        case 84:
            throw new Exception("Alternate Scaling Required");
        case 86:
            if (nc < 6) throw new Exception("Insufficient constants");
            if (x < c[0]) return c[2] * x + c[3];
            if (x > c[1]) return Math.exp(c[4] * x + c[5]);
            // logarithmic interpolation
            double x0 = c[0], y0 = c[2] * c[0] + c[3];
            double x1 = c[1], y1 = Math.exp(c[4] * c[1] + c[5]);
            double log_result = Math.log(y1 / y0) * (x - x0) / (x1 - x0) + Math.log(y0);
            return Math.exp(log_result);
        case 88:
            if (nc < 6) throw new Exception("Insufficient constants");
            return (c[0] + x * (c[1] + x * c[2])) / (1.0 + x * (c[3] + x * (c[4] + x * c[5])));
        case 90:
            throw new Exception("Multifunction table not supported");
        case 201:
            throw new Exception("Sub-transform of 90 not supported");
        default:
            throw new Exception("Common transform " + cIndex + " not found");
        }
    }

    // ---- Binary Search ----

    static double binarySearch(int cIndex, double[] constants, double tol,
                               double pul, double puu, double xx) throws Exception {
        double toler = tol;
        if (toler > 0.0001) toler = 0.0001;
        double pulow = pul, puupr = puu;
        double cuupr = commonScale(cIndex, constants, puupr) - xx;
        double culow = commonScale(cIndex, constants, pulow) - xx;
        double pumid, cumid, diff, d1, d2, d3;
        int iter = 0;

        // Phase 1: find sign-change interval
        while (true) {
            if (((cuupr > 0) && (culow > 0)) || ((cuupr < 0) && (culow < 0))) {
                pumid = (pulow + puupr) / 2.0;
                cumid = commonScale(cIndex, constants, pumid) - xx;
                if ((Math.abs(puupr) < FLT_MAX / 2.0) && (Math.abs(pulow) < FLT_MAX / 2.0)) {
                    diff = puupr - pulow;
                    if (Math.abs(diff) <= toler)
                        throw new Exception("Interval not found");
                }
                if ((pumid == pulow) || (pumid == puupr))
                    throw new Exception("Interval not found");
                if (((cuupr > 0) && (cumid < 0)) || ((cuupr < 0) && (cumid > 0))) {
                    puupr = pumid;
                    cuupr = cumid;
                    break;
                }
                d1 = Math.abs(cumid);
                d2 = Math.abs(culow);
                d3 = Math.abs(cuupr);
                if ((d3 > d1) && (d3 >= d2)) {
                    puupr = pumid;
                    cuupr = cumid;
                } else if ((d2 > d1) && (d2 >= d3)) {
                    pulow = pumid;
                    culow = cumid;
                } else {
                    throw new Exception("Interval not found");
                }
                iter++;
                if (iter >= MAX_SEARCH)
                    throw new Exception("Interval not found");
            } else {
                break;
            }
        }

        // Phase 2: bisect to convergence
        iter = 0;
        while (true) {
            pumid = (pulow + puupr) / 2.0;
            if ((Math.abs(puupr) < FLT_MAX / 2.0) || (Math.abs(pulow) < FLT_MAX / 2.0)) {
                diff = puupr - pulow;
                if (Math.abs(diff) <= toler)
                    return pumid;
            }
            if ((pumid == pulow) || (pumid == puupr))
                return pumid;
            cuupr = commonScale(cIndex, constants, puupr);
            cumid = commonScale(cIndex, constants, pumid);
            if (cumid == xx)
                return pumid;
            else if (cumid > xx && cuupr >= xx)
                puupr = pumid;
            else if (cumid < xx && cuupr <= xx)
                puupr = pumid;
            else if (cumid < xx && cuupr >= xx)
                pulow = pumid;
            else if (cumid > xx && cuupr <= xx)
                pulow = pumid;
            iter++;
            if (iter >= MAX_SEARCH)
                throw new Exception("Too many iterations");
        }
    }

    // ---- Root Bisection ----

    static double rootBisection(int cIndex, double[] constants,
                                double desired, double x1, double x2, double xacc) throws Exception {
        double f = commonScale(cIndex, constants, x1) - desired;
        double fmid = commonScale(cIndex, constants, x2) - desired;
        double dx, xmid, rtb;

        if ((f * fmid) >= 0.0)
            throw new Exception("Value not bracketed");

        if (f < 0.0) {
            rtb = x1;
            dx = x2 - x1;
        } else {
            dx = x1 - x2;
            rtb = x2;
        }

        for (int j = 1; j <= MAX_BISECT; j++) {
            dx *= 0.5;
            xmid = rtb + dx;
            fmid = commonScale(cIndex, constants, xmid) - desired;
            if (fmid <= 0.0) rtb = xmid;
            if ((Math.abs(dx) < xacc) || (fmid == 0.0))
                return rtb;
        }
        return rtb;
    }

    // ---- Common Unscale (common -> primary) ----

    static double commonUnscale(int cIndex, double[] c, int pIndex, double xx) throws Exception {
        double state, tmpx, tol;
        double pulow, puupr;

        int limIdx = pIndex / 2;
        if (limIdx < lowlim.length) {
            pulow = lowlim[limIdx];
            puupr = uprlim[limIdx];
        } else {
            pulow = 0.0;
            puupr = 0.0;
        }

        switch (cIndex) {
        case 0:
            return xx;
        case 2:
        case 40:
            return (xx - c[2]) * c[1] / c[0];
        case 4:
            return xx * c[1] + c[0];
        case 6:
            return xx * c[1] / c[0];
        case 8:
            if ((c[0] - c[1] * (xx - c[3])) == 0)
                throw new Exception("Division by zero");
            return c[2] * (xx - c[3]) / (c[0] - c[1] * (xx - c[3]));
        case 10:
            if (((xx - c[2]) * c[0]) == 0)
                throw new Exception("Division by zero");
            return c[1] / ((xx - c[2]) * c[0]);
        case 12:
            if ((c[0] == 0.) && (c[1] == 0.) && (c[2] != 0.)) {
                double aa = c[2], bb = c[3], cc = c[4] - xx;
                double discr = bb * bb - 4. * aa * cc;
                if (discr < 0)
                    throw new Exception("Negative discriminant");
                return (-bb + Math.sqrt(discr)) / (2. * aa);
            }
            tol = puupr / TOLDIV;
            return binarySearch(cIndex, c, tol, pulow, puupr, xx);
        case 20:
            if (c[2] == 0.0) tmpx = xx;
            else tmpx = xx - c[2];
            double a20 = -4 * tmpx * c[0] * c[1] + 1;
            if (a20 < 0 || tmpx == 0)
                throw new Exception("Domain error");
            double b20 = (float) (((1 - 2 * tmpx * c[0] * c[1] + Math.sqrt(a20))
                / (2 * tmpx * c[0] * c[0])) / 0.43429);
            return (float) (Math.exp(b20));
        case 28:
            if ((c[0] * (xx - c[3])) == 0)
                throw new Exception("Division by zero");
            return c[2] / (c[0] * (xx - c[3])) - (c[1] / c[0]);
        case 32:
            double a32 = (xx - c[2]) / c[1];
            return (float) (Math.exp(a32) / c[0] - c[3]);
        case 34:
            if ((c[0] - c[2] * xx) == 0)
                throw new Exception("Division by zero");
            return (c[3] * xx - c[1]) / (c[0] - c[2] * xx);
        case 36:
            return ((xx - c[2]) / c[1]) * ((xx - c[2]) / c[1]) - c[0];
        case 38:
            if (xx == 7.6E5) return c[5];
            if ((xx > 7.6E5) || (xx <= 0))
                throw new Exception("Value out of range");
            if (((c[2] == 0) && (c[3] == 0) && (c[4] == 0)) && (c[1] != 0))
                return (Math.log(xx) / Math.log(10) - c[0]) / c[1];
            if (pIndex == 16) {
                pulow = 1.00001 * c[5];
                puupr = 15.0;
            } else {
                if (c[5] == 0) pulow = 0.00001;
                else pulow = 1.00001 * c[5];
            }
            tol = (float) (puupr / TOLDIV);
            return rootBisection(cIndex, c, xx, pulow, puupr, tol);
        case 50:
            return (float) (c[1] * Math.cos(xx / c[0]));
        case 56:
            throw new Exception("Table lookup not supported");
        case 58:
            throw new Exception("Table lookup not supported");
        case 60:
            return xx;
        case 62:
            if ((c[0] == 0.0) || (c[1] == 0.0))
                throw new Exception("Zero constant");
            tmpx = xx / c[1] - c[2];
            if (tmpx < 0)
                throw new Exception("Logarithm of negative value");
            return c[0] * Math.log(tmpx) / Math.log(10);
        case 64:
            pulow = -1.75489;
            puupr = 5.05;
            tol = (double) ((puupr - pulow) / TOLDIV);
            if ((xx < 0) || (xx > commonScale(cIndex, c, puupr)))
                throw new Exception("Value out of range");
            return binarySearch(cIndex, c, tol, pulow, puupr, xx);
        case 66:
            return ((Math.log((xx - c[3]) / c[0]) / 0.6931471805599453) - c[2]) / c[1];
        case 68:
            puupr = 0.0;
            tol = (float) ((puupr - pulow) / TOLDIV);
            if ((xx < 0) || (xx > commonScale(cIndex, c, pulow)))
                throw new Exception("Value out of range");
            return binarySearch(cIndex, c, tol, pulow, puupr, xx);
        case 74: {
            double a = xx * c[5] - c[2];
            double b = xx * c[4] - c[1];
            double g = xx * c[3] - c[0];
            double d = b * b - 4 * a * g;
            if (d < 0)
                throw new Exception("Negative discriminant");
            d = Math.sqrt(d);
            tmpx = (-b + d) / (2 * a);
            if ((tmpx >= pulow) && (tmpx <= puupr))
                state = tmpx;
            else
                state = (-b - d) / (2 * a);
            if ((state < pulow) || (state > puupr))
                throw new Exception("Value out of range");
            return state;
        }
        case 76: {
            double a = c[1] * Math.pow(c[0], c[2]);
            double b = c[3] * c[4];
            if ((c[1] > 0) && (b > 0)) {
                if (xx < a) return Math.pow(xx / c[1], 1.0 / c[2]);
                return (Math.log(xx / c[3]) - c[5]) / c[4];
            }
            if ((c[1] < 0) && (b < 0)) {
                if (xx >= a) return Math.pow(xx / c[1], 1.0 / c[2]);
                return (Math.log(xx / c[3]) - c[5]) / c[4];
            }
            if ((c[1] > 0) && (b < 0)) {
                if (xx < a) return Math.pow(xx / c[1], 1.0 / c[2]);
                throw new Exception("Value out of range");
            }
            if (xx >= a) return Math.pow(xx / c[1], 1.0 / c[2]);
            throw new Exception("Value out of range");
        }
        case 78:
            return (Math.log((xx - c[3]) / c[0]) / Math.log(10) - c[2]) / c[1];
        case 80:
            return xx;
        case 86: {
            double x_0 = c[0], y_0 = c[2] * c[0] + c[3];
            double x_1 = c[1], y_1 = Math.exp(c[4] * c[1] + c[5]);
            if (xx > y_0) return (xx - c[3]) / c[2];
            if (xx < y_1) return (Math.log(xx) - c[5]) / c[4];
            return (x_1 - x_0) * Math.log(xx / y_0) / Math.log(y_1 / y_0) + x_0;
        }
        default:
            tol = puupr / TOLDIV;
            return binarySearch(cIndex, c, tol, pulow, puupr, xx);
        }
    }
}
